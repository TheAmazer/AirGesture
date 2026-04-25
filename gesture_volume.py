"""
gesture_volume.py

Robust state-machine gesture control:
- IDLE: fist/rest, no actions
- NAVIGATING: open palm, swipe left/right
- ADJUSTING: index-only, circular rotation for volume
"""

import math
import platform
import subprocess
import time
from collections import deque
from enum import Enum, auto
from typing import Callable, Deque, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HandState(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    ADJUSTING = auto()


threshold_config: Dict[str, float] = {
    # --- Swipe detection ---
    "linearity_threshold": 0.60,             # slightly stricter (was 0.56)
    "swipe_max_duration_s": 0.40,            # allow slightly slower swipes (was 0.30)
    "swipe_min_velocity_px_s": 280.0,        # easier to trigger (was 490)
    "swipe_min_displacement_px": 55.0,       # NEW: minimum pixel travel
    "swipe_direction_consistency": 0.72,     # NEW: fraction of steps in dominant dir
    # --- Continuous rotation tracker (volume) ---
    "rotation_step_deg": 30.0,              # raised from 22: reduces jitter-driven oscillation
    "rotation_min_radius_px": 22.0,         # NEW: ignore angle when tip too near wrist
    # --- Finger / pose detection ---
    "finger_open_scale": 1.8,
    "finger_hysteresis": 0.15,
    "thumb_tucked_scale": 1.10,
    "adjust_enter_extension": 2.1,
    "adjust_exit_extension": 1.6,
    "index_fold_strict_scale": 1.55,        # relaxed: bent-not-fisted fingers sit at ~1.3-1.7x MCP dist
    # --- State machine ---
    "deadzone_px": 5.6,
    "state_debounce_frames": 10,            # raised: exit ADJUSTING only after 10 steady frames (~333ms)
    "adjust_enter_frames": 3,               # NEW: consecutive frames needed to enter ADJUSTING
    "adjust_lost_hold_frames": 5,           # raised: hold longer when hand briefly leaves frame
    # --- Legacy keys (kept for is_finger_open compatibility) ---
    "circular_consistency_threshold": 0.56,
    "rotation_min_duration_s": 0.50,
    "rotation_max_duration_s": 1.00,
    "rotation_min_angle_deg": 84.0,
}


def get_mediapipe_legacy_hands():
    if hasattr(mp, "solutions"):
        return mp.solutions.hands, mp.solutions.drawing_utils
    raise RuntimeError(
        "This environment lacks `mp.solutions`. Use Python 3.11 + mediapipe==0.10.14."
    )


class MovingAverage2D:
    def __init__(self, window: int = 5):
        self._x: Deque[float] = deque(maxlen=window)
        self._y: Deque[float] = deque(maxlen=window)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        self._x.append(x)
        self._y.append(y)
        return sum(self._x) / len(self._x), sum(self._y) / len(self._y)


class WeightedLandmarkSmoother:
    def __init__(self, window: int = 5):
        self.window = window
        self.weights = np.arange(1, window + 1, dtype=np.float64)
        self.weights = self.weights / np.sum(self.weights)
        self.history: Deque[np.ndarray] = deque(maxlen=window)

    def update(self, hand_landmarks) -> np.ndarray:
        frame_pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float64)
        self.history.append(frame_pts)
        stack = np.stack(list(self.history), axis=0)
        w = self.weights[-len(self.history):]
        w = w / np.sum(w)
        return np.tensordot(w, stack, axes=(0, 0))


class ContinuousRotationTracker:
    """
    Tracks the signed angle of the index-fingertip around the wrist landmark
    each frame and fires discrete volume steps as the angle accumulates.
      CW  (+steps) -> Volume UP
      CCW (-steps) -> Volume DOWN
    """

    def __init__(self, step_angle_deg: float = 22.0, min_radius_px: float = 22.0):
        self.step_rad = math.radians(step_angle_deg)
        self.min_radius_px = min_radius_px
        self._accumulated = 0.0
        self._last_angle: Optional[float] = None

    def reset(self) -> None:
        self._accumulated = 0.0
        self._last_angle = None

    def update(self, tip_x: float, tip_y: float, ref_x: float, ref_y: float) -> int:
        """
        Feed one frame's index-tip and wrist positions (pixels).
        Returns step count: +N = N CW steps (vol up), -N = CCW (vol down), 0 = no step.
        """
        dx = tip_x - ref_x
        dy = tip_y - ref_y
        radius = math.hypot(dx, dy)
        if radius < self.min_radius_px:
            return 0  # too close to reference — angle is unreliable
        angle = math.atan2(dy, dx)  # screen coords: CW = positive angle accumulation
        steps = 0
        if self._last_angle is not None:
            delta = wrapped_angle_delta(self._last_angle, angle)
            self._accumulated += delta
            while self._accumulated >= self.step_rad:
                self._accumulated -= self.step_rad
                steps += 1
            while self._accumulated <= -self.step_rad:
                self._accumulated += self.step_rad
                steps -= 1
        self._last_angle = angle
        return steps

    @property
    def accumulated_deg(self) -> float:
        return math.degrees(self._accumulated)


class GestureManager:
    def __init__(self):
        self.current_state = HandState.IDLE
        self.pending_state: Optional[HandState] = None
        self.state_confidence_counter = 0
        self.lost_frames = 0
        self.exit_debounce_frames = int(threshold_config["state_debounce_frames"])
        self.enter_debounce_frames = int(threshold_config["adjust_enter_frames"])

    def is_locked(self) -> bool:
        return self.current_state == HandState.ADJUSTING

    def update_state(
        self,
        hand_present: bool,
        finger_up: Optional[Dict[str, bool]],
    ) -> Tuple[HandState, bool]:
        """
        Returns: (state, changed)
        """
        changed = False
        if hand_present:
            self.lost_frames = 0
            fingers = finger_up or {}
            count = sum(1 for k, v in fingers.items() if v and not k.startswith("_"))
            index_up = fingers.get("index", False)
            middle_up = fingers.get("middle", False)

            # Priority-based override (logic-first):
            # 1) strict pointer pose -> ADJUSTING (immediate)
            # 2) open palm -> NAVIGATING
            # 3) otherwise -> IDLE
            if fingers.get("_strict_index_only", False):
                proposed = HandState.ADJUSTING
            elif count >= 4:
                proposed = HandState.NAVIGATING
            else:
                proposed = HandState.IDLE

            # Debounced entry to ADJUSTING: require N consecutive frames to avoid flicker.
            if proposed == HandState.ADJUSTING and self.current_state != HandState.ADJUSTING:
                if self.pending_state == HandState.ADJUSTING:
                    self.state_confidence_counter += 1
                else:
                    self.pending_state = HandState.ADJUSTING
                    self.state_confidence_counter = 1

                if self.state_confidence_counter >= self.enter_debounce_frames:
                    self.current_state = HandState.ADJUSTING
                    self.pending_state = None
                    self.state_confidence_counter = 0
                    return self.current_state, True
                # Not enough consecutive frames yet — stay in current state.
                return self.current_state, False

            # Debounced exits from ADJUSTING to reduce flicker.
            if self.current_state == HandState.ADJUSTING and proposed != HandState.ADJUSTING:
                if self.pending_state == proposed:
                    self.state_confidence_counter += 1
                else:
                    self.pending_state = proposed
                    self.state_confidence_counter = 1

                if self.state_confidence_counter >= self.exit_debounce_frames:
                    self.current_state = proposed
                    self.pending_state = None
                    self.state_confidence_counter = 0
                    changed = True
                return self.current_state, changed

            if proposed != self.current_state:
                self.current_state = proposed
                self.pending_state = None
                self.state_confidence_counter = 0
                changed = True
            else:
                self.pending_state = None
                self.state_confidence_counter = 0
            return self.current_state, changed

        # Hand lost: hold ADJUSTING for a few frames to avoid flicker.
        if self.current_state == HandState.ADJUSTING and self.lost_frames < int(threshold_config["adjust_lost_hold_frames"]):
            self.lost_frames += 1
            return self.current_state, False

        if self.current_state != HandState.IDLE:
            changed = True
        self.current_state = HandState.IDLE
        self.pending_state = None
        self.state_confidence_counter = 0
        return self.current_state, changed


class VolumeController:
    def __init__(self, step_percent: int = 4):
        self.system = platform.system()
        self.step_percent = max(1, min(20, step_percent))
        self._windows_volume = None
        self._is_windows_ready = False
        self._is_mac = self.system == "Darwin"
        if self.system == "Windows":
            self._setup_windows()

    def _setup_windows(self) -> None:
        try:
            from pycaw.pycaw import AudioUtilities  # type: ignore

            devices = AudioUtilities.GetSpeakers()
            if hasattr(devices, "EndpointVolume"):
                self._windows_volume = devices.EndpointVolume
            else:
                from ctypes import POINTER, cast

                from comtypes import CLSCTX_ALL  # type: ignore
                from pycaw.pycaw import IAudioEndpointVolume  # type: ignore

                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._windows_volume = cast(interface, POINTER(IAudioEndpointVolume))
            self._is_windows_ready = True
        except Exception as exc:
            print(f"[WARN] Windows volume setup failed (pycaw/comtypes): {exc}")
            print("[WARN] Volume changes will be disabled on Windows.")

    def _mac_get_volume(self) -> int:
        try:
            out = subprocess.check_output(
                ["osascript", "-e", "output volume of (get volume settings)"], text=True
            ).strip()
            return int(out)
        except Exception:
            return 50

    def _mac_set_volume(self, value: int) -> None:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {max(0, min(100, int(value)))}"],
            check=False,
        )

    def change(self, delta_percent: int) -> Optional[int]:
        if self.system == "Windows" and self._is_windows_ready and self._windows_volume is not None:
            current = float(self._windows_volume.GetMasterVolumeLevelScalar())
            new_val = max(0.0, min(1.0, current + delta_percent / 100.0))
            self._windows_volume.SetMasterVolumeLevelScalar(new_val, None)
            return int(new_val * 100)
        if self._is_mac:
            current = self._mac_get_volume()
            new_val = max(0, min(100, current + delta_percent))
            self._mac_set_volume(new_val)
            return new_val
        return None

    def increase(self) -> Optional[int]:
        return self.change(+self.step_percent)

    def decrease(self) -> Optional[int]:
        return self.change(-self.step_percent)


def _dist(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def get_index_extension_ratio(smoothed_points: np.ndarray) -> float:
    hand_scale = _dist(smoothed_points, 0, 9) + 1e-6
    tip_to_wrist = _dist(smoothed_points, 8, 0)
    return float(tip_to_wrist / hand_scale)


def get_finger_up_map_binary(smoothed_points: np.ndarray) -> Dict[str, bool]:
    """
    Finger is up if Tip-to-Wrist distance > middle-joint-to-Wrist distance.
    Thumb uses IP as middle joint.
    """
    wrist = 0
    finger_pairs = {
        "thumb": (4, 3),   # tip, IP
        "index": (8, 6),   # tip, PIP
        "middle": (12, 10),
        "ring": (16, 14),
        "pinky": (20, 18),
    }
    up = {}
    for name, (tip, mid) in finger_pairs.items():
        up[name] = _dist(smoothed_points, tip, wrist) > _dist(smoothed_points, mid, wrist)

    # Middle-finger guard for true pointer pose.
    middle_guard_folded = _dist(smoothed_points, 12, wrist) < _dist(smoothed_points, 9, wrist)
    if not middle_guard_folded:
        up["middle"] = True
    return up


def is_index_only_pose(smoothed_points: np.ndarray) -> bool:
    """
    Strict check for the pointer pose:
      - Index tip clearly extended past its PIP joint.
      - Middle, ring, pinky tips must be within index_fold_strict_scale * MCP-to-wrist,
        meaning they haven't cleared the knuckle line.
    Prevents false ADJUSTING state from slightly bent non-index fingers.
    """
    wrist = 0
    strict = threshold_config["index_fold_strict_scale"]
    # Index must be clearly extended.
    if not (_dist(smoothed_points, 8, wrist) > _dist(smoothed_points, 6, wrist) * 1.05):
        return False
    # Middle, ring, pinky must be folded below the MCP line.
    for tip_idx, mcp_idx in ((12, 9), (16, 13), (20, 17)):
        if _dist(smoothed_points, tip_idx, wrist) > _dist(smoothed_points, mcp_idx, wrist) * strict:
            return False
    return True


def is_finger_open(
    smoothed_points: np.ndarray,
    finger_index: int,
    previous_state: Optional[bool] = None,
) -> bool:
    """
    finger_index: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    """
    wrist = 0
    middle_mcp = 9
    hand_scale = _dist(smoothed_points, wrist, middle_mcp) + 1e-6
    open_threshold = threshold_config["finger_open_scale"] * hand_scale
    close_threshold = open_threshold * (1.0 - threshold_config["finger_hysteresis"])

    if finger_index == 0:
        thumb_tip = 4
        pinky_mcp = 17
        thumb_ref = _dist(smoothed_points, thumb_tip, pinky_mcp)
        thumb_open_threshold = threshold_config["thumb_tucked_scale"] * hand_scale
        thumb_close_threshold = thumb_open_threshold * (1.0 - threshold_config["finger_hysteresis"])
        if previous_state is True:
            return thumb_ref > thumb_close_threshold
        if previous_state is False:
            return thumb_ref > thumb_open_threshold
        return thumb_ref > thumb_open_threshold

    tip_idx_map = {1: 8, 2: 12, 3: 16, 4: 20}
    tip_idx = tip_idx_map[finger_index]
    tip_to_wrist = _dist(smoothed_points, tip_idx, wrist)
    if previous_state is True:
        return tip_to_wrist > close_threshold
    if previous_state is False:
        return tip_to_wrist > open_threshold
    return tip_to_wrist > open_threshold


def classify_hand_state(finger_open: Dict[str, bool]) -> HandState:
    extended_count = sum(1 for v in finger_open.values() if v)
    if finger_open["index"] and extended_count == 1:
        return HandState.ADJUSTING
    if extended_count >= 4:
        return HandState.NAVIGATING
    return HandState.IDLE


def wrapped_angle_delta(a1: float, a2: float) -> float:
    d = a2 - a1
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def analyze_trajectory(buffer: Deque[Tuple[float, float, float]]) -> Dict[str, float]:
    metrics = {
        "linearity_score": 0.0,
        "radius_variance": 0.0,
        "circular_consistency": 0.0,
        "duration": 0.0,
        "velocity": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "total_angle": 0.0,
        "dir_consistency_x": 0.0,
        "dir_consistency_y": 0.0,
    }
    if len(buffer) < 10:
        return metrics

    arr = np.array(buffer, dtype=np.float64)
    xy = arr[:, :2]
    ts = arr[:, 2]
    duration = float(ts[-1] - ts[0])
    metrics["duration"] = duration
    if duration <= 0.0:
        return metrics

    centroid = np.mean(xy, axis=0)
    radii = np.linalg.norm(xy - centroid, axis=1)
    mean_r = float(np.mean(radii))
    if mean_r < 1e-6:
        return metrics
    radius_cv = float(np.std(radii) / (mean_r + 1e-6))
    circular_consistency = max(0.0, 1.0 - radius_cv)
    metrics["radius_variance"] = float(np.std(radii) ** 2)
    metrics["circular_consistency"] = circular_consistency

    start = xy[0]
    end = xy[-1]
    displacement = end - start
    metrics["dx"] = float(displacement[0])
    metrics["dy"] = float(displacement[1])
    disp_norm = float(np.linalg.norm(displacement))
    segments = np.diff(xy, axis=0)
    path_len = float(np.sum(np.linalg.norm(segments, axis=1)) + 1e-6)
    straightness = min(1.0, disp_norm / path_len)

    line_unit = displacement / (disp_norm + 1e-6)
    projected = np.dot(xy - start, line_unit)
    closest = start + np.outer(projected, line_unit)
    deviation = np.linalg.norm(xy - closest, axis=1)
    deviation_score = max(0.0, 1.0 - float(np.mean(deviation) / (disp_norm + 1e-6)))
    linearity_score = 0.5 * (straightness + deviation_score)
    metrics["linearity_score"] = linearity_score

    velocity = disp_norm / duration  # px / s
    metrics["velocity"] = velocity

    angles = np.arctan2(-(xy[:, 1] - centroid[1]), (xy[:, 0] - centroid[0]))
    d = np.diff(angles)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    metrics["total_angle"] = float(np.sum(d))

    # Directional consistency: fraction of per-frame steps moving in the dominant direction.
    if len(xy) >= 3:
        steps_x = np.diff(xy[:, 0])
        steps_y = np.diff(xy[:, 1])
        n = max(len(steps_x), 1)
        dom_x = max(np.sum(steps_x > 0), np.sum(steps_x < 0))
        dom_y = max(np.sum(steps_y > 0), np.sum(steps_y < 0))
        metrics["dir_consistency_x"] = float(dom_x / n)
        metrics["dir_consistency_y"] = float(dom_y / n)

    return metrics


def classify_trajectory(buffer: Deque[Tuple[float, float, float]]) -> str:
    """
    Returns: SWIPE_LEFT, SWIPE_RIGHT, or NONE.
    Rotation (CW/CCW) is handled continuously by ContinuousRotationTracker.
    """
    metrics = analyze_trajectory(buffer)
    duration = metrics["duration"]
    linearity_score = metrics["linearity_score"]
    velocity = metrics["velocity"]
    dx = metrics["dx"]
    dy = metrics["dy"]
    dir_x = metrics["dir_consistency_x"]
    disp = math.hypot(dx, dy)

    if (
        linearity_score >= threshold_config["linearity_threshold"]
        and duration <= threshold_config["swipe_max_duration_s"]
        and velocity >= threshold_config["swipe_min_velocity_px_s"]
        and disp >= threshold_config["swipe_min_displacement_px"]
        and abs(dx) > abs(dy) * 1.4          # clearly horizontal
        and dir_x >= threshold_config["swipe_direction_consistency"]
    ):
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"

    return "NONE"


def moved_enough(prev: Optional[Tuple[float, float]], current: Tuple[float, float], deadzone_px: float) -> bool:
    if prev is None:
        return True
    return math.hypot(current[0] - prev[0], current[1] - prev[1]) >= deadzone_px


def draw_debug_info(
    frame,
    trajectory: Deque[Tuple[float, float, float]],
    state: HandState,
    action_text: str,
    fps: float,
    finger_count: int,
    metrics: Dict[str, float],
    trajectory_label: str,
    rotation_lock: bool,
    accumulated_rot_deg: float = 0.0,
) -> None:
    if trajectory:
        pts = np.array([(int(x), int(y)) for x, y, _ in list(trajectory)[-20:]], dtype=np.int32)
        if len(pts) >= 2:
            if trajectory_label.startswith("SWIPE"):
                color = (255, 0, 0)      # Blue (BGR)
            elif state == HandState.ADJUSTING:
                color = (0, 255, 0)      # Green while rotating
            else:
                color = (0, 0, 255)      # Red
            cv2.polylines(frame, [pts], False, color, 2)

    cv2.putText(frame, f"State: {state.name}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 220, 50), 2)
    cv2.putText(frame, f"Action: {action_text}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.putText(frame, f"Linearity: {metrics['linearity_score']:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
    cv2.putText(frame, f"Dir Consistency: {metrics['dir_consistency_x']:.2f}", (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
    cv2.putText(frame, f"Rot accum: {accumulated_rot_deg:+.1f} deg", (10, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 180, 255), 2)
    cv2.putText(frame, f"Rot lock: {'ON' if rotation_lock else 'OFF'}", (10, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)
    cv2.putText(frame, "q = quit", (10, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 2)


def main() -> None:
    target_fps = 30.0
    frame_interval = 1.0 / target_fps
    deadzone_px = threshold_config["deadzone_px"]
    adjust_points: Deque[Tuple[float, float, float]] = deque(maxlen=40)
    nav_points: Deque[Tuple[float, float, float]] = deque(maxlen=25)
    smoother = MovingAverage2D(window=5)
    last_adjust_point: Optional[Tuple[float, float]] = None
    last_nav_point: Optional[Tuple[float, float]] = None
    last_nav_trigger = 0.0
    last_action_text = "None"
    current_state = HandState.IDLE
    prev_state = HandState.IDLE
    finger_count = 0
    last_metrics = {
        "linearity_score": 0.0,
        "radius_variance": 0.0,
        "circular_consistency": 0.0,
        "duration": 0.0,
        "velocity": 0.0,
        "dx": 0.0,
        "dy": 0.0,
        "total_angle": 0.0,
        "dir_consistency_x": 0.0,
        "dir_consistency_y": 0.0,
    }
    trajectory_label = "NONE"
    rot_tracker = ContinuousRotationTracker(
        step_angle_deg=threshold_config["rotation_step_deg"],
        min_radius_px=threshold_config["rotation_min_radius_px"],
    )
    landmark_smoother = WeightedLandmarkSmoother(window=5)
    gesture_manager = GestureManager()
    finger_state_memory: Dict[str, Optional[bool]] = {
        "thumb": None,
        "index": None,
        "middle": None,
        "ring": None,
        "pinky": None,
    }

    vol = VolumeController(step_percent=4)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_hands, mp_draw = get_mediapipe_legacy_hands()
    swipe_gesture_mapper: Dict[str, str] = {
        "SWIPE_LEFT": "Swipe Left",
        "SWIPE_RIGHT": "Swipe Right",
    }

    fps_t0 = time.time()
    frame_counter = 0
    shown_fps = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            t_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            current_state = HandState.IDLE
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                smoothed_landmarks = landmark_smoother.update(hand)
                wrist_x = float(smoothed_landmarks[0, 0] * w)
                wrist_y = float(smoothed_landmarks[0, 1] * h)
                finger_up = get_finger_up_map_binary(smoothed_landmarks)
                finger_up["_strict_index_only"] = is_index_only_pose(smoothed_landmarks)
                finger_state_memory.update({k: v for k, v in finger_up.items() if not k.startswith("_")})
                current_state, _ = gesture_manager.update_state(
                    hand_present=True,
                    finger_up=finger_up,
                )
                # Exclude thumb from count: it often false-positives in pointer pose
                finger_count = sum(1 for k, v in finger_up.items()
                                   if v and k not in ("thumb",) and not k.startswith("_"))

                tip_x = float(smoothed_landmarks[8, 0] * w)
                tip_y = float(smoothed_landmarks[8, 1] * h)
                smoothed = smoother.update(tip_x, tip_y)
                cv2.circle(frame, (int(smoothed[0]), int(smoothed[1])), 7, (0, 255, 255), -1)

                # State classification happens first; only then compute state-specific gestures.
                now_point_t = time.time()
                if current_state == HandState.ADJUSTING:
                    nav_points.clear()
                    last_nav_point = None
                    if moved_enough(last_adjust_point, smoothed, deadzone_px):
                        adjust_points.append((smoothed[0], smoothed[1], now_point_t))
                        last_adjust_point = smoothed
                    # Continuous rotation -> fire volume steps immediately
                    steps = rot_tracker.update(smoothed[0], smoothed[1], wrist_x, wrist_y)
                    for _ in range(abs(steps)):
                        if steps > 0:
                            new_vol = vol.increase()
                            last_action_text = f"CW -> Vol UP ({new_vol if new_vol is not None else 'n/a'}%)"
                        else:
                            new_vol = vol.decrease()
                            last_action_text = f"CCW -> Vol DOWN ({new_vol if new_vol is not None else 'n/a'}%)"
                elif current_state == HandState.NAVIGATING:
                    adjust_points.clear()
                    last_adjust_point = None
                    if moved_enough(last_nav_point, smoothed, deadzone_px):
                        nav_points.append((smoothed[0], smoothed[1], now_point_t))
                        last_nav_point = smoothed
                else:
                    adjust_points.clear()
                    nav_points.clear()
                    last_adjust_point = None
                    last_nav_point = None
            else:
                # Hand left frame: immediately clear to prevent ghost gestures.
                current_state, _ = gesture_manager.update_state(
                    hand_present=False,
                    finger_up=None,
                )
                if current_state == HandState.IDLE:
                    adjust_points.clear()
                    nav_points.clear()
                    last_adjust_point = None
                    last_nav_point = None
                    trajectory_label = "NONE"
                    finger_count = 0
                landmark_smoother.history.clear()
                finger_state_memory = {
                    "thumb": None,
                    "index": None,
                    "middle": None,
                    "ring": None,
                    "pinky": None,
                }

            now = time.time()
            if (
                current_state == HandState.NAVIGATING
                and not gesture_manager.is_locked()
                and now - last_nav_trigger >= 0.20
            ):
                last_metrics = analyze_trajectory(nav_points)
                traj = classify_trajectory(nav_points)
                trajectory_label = traj
                if traj in swipe_gesture_mapper:
                    last_action_text = swipe_gesture_mapper[traj]
                    last_nav_trigger = now
                    nav_points.clear()
                    last_nav_point = None
            elif current_state == HandState.ADJUSTING:
                trajectory_label = "ADJUSTING"
            elif current_state == HandState.IDLE:
                trajectory_label = "NONE"

            if current_state != prev_state:
                print(f"State changed: {prev_state.name} -> {current_state.name}")
                if prev_state == HandState.ADJUSTING:
                    rot_tracker.reset()
                prev_state = current_state

            frame_counter += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                shown_fps = frame_counter / dt
                frame_counter = 0
                fps_t0 = time.time()

            active_trajectory = adjust_points if current_state == HandState.ADJUSTING else nav_points
            draw_debug_info(
                frame=frame,
                trajectory=active_trajectory,
                state=current_state,
                action_text=last_action_text,
                fps=shown_fps,
                finger_count=finger_count,
                metrics=last_metrics,
                trajectory_label=trajectory_label,
                rotation_lock=gesture_manager.is_locked(),
                accumulated_rot_deg=rot_tracker.accumulated_deg,
            )

            cv2.imshow("AirGesture State Machine", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
