"""
gesture_volume_classifier_app.py

BMW-style gesture volume controller using:
- MediaPipe Hands for landmarks
- KeyPointClassifier from hand-gesture-recognition-using-mediapipe-main
- Pycaw for Windows volume control

State mapping:
- Pointer -> ADJUSTING
- Open -> NAVIGATING
- Fist/Close -> IDLE
"""

import csv
import math
import os
import platform
import subprocess
import sys
import time
from collections import deque
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_REPO = os.path.join(PROJECT_DIR, "hand-gesture-recognition-using-mediapipe-main")
sys.path.append(CLASSIFIER_REPO)

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier as KeypointClassifier  # noqa: E402


class HandState(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    ADJUSTING = auto()


class VolumeController:
    def __init__(self, step_percent: int = 3):
        self.system = platform.system()
        self.step_percent = step_percent
        self._windows_volume = None
        self._is_windows_ready = False
        if self.system == "Windows":
            self._setup_windows()

    def _setup_windows(self) -> None:
        try:
            from pycaw.pycaw import AudioUtilities  # type: ignore

            device = AudioUtilities.GetSpeakers()
            self._windows_volume = (
                device.EndpointVolume if hasattr(device, "EndpointVolume") else None
            )
            self._is_windows_ready = self._windows_volume is not None
        except Exception as exc:
            print(f"[WARN] Volume init failed: {exc}")

    def change(self, delta_percent: int) -> Optional[int]:
        if self.system == "Windows" and self._is_windows_ready:
            current = float(self._windows_volume.GetMasterVolumeLevelScalar())
            new_val = max(0.0, min(1.0, current + delta_percent / 100.0))
            self._windows_volume.SetMasterVolumeLevelScalar(new_val, None)
            return int(new_val * 100)

        if self.system == "Darwin":
            try:
                out = subprocess.check_output(
                    ["osascript", "-e", "output volume of (get volume settings)"],
                    text=True,
                ).strip()
                current = int(out)
            except Exception:
                current = 50
            new_val = max(0, min(100, current + delta_percent))
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {new_val}"], check=False
            )
            return new_val
        return None

    def increase(self) -> Optional[int]:
        return self.change(self.step_percent)

    def decrease(self) -> Optional[int]:
        return self.change(-self.step_percent)


def calc_landmark_list(image, landmarks) -> List[List[int]]:
    image_width, image_height = image.shape[1], image.shape[0]
    points = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        points.append([x, y])
    return points


def pre_process_landmark(landmark_list: List[List[int]]) -> List[float]:
    base_x, base_y = landmark_list[0]
    temp = [[p[0] - base_x, p[1] - base_y] for p in landmark_list]
    flat = [coord for xy in temp for coord in xy]
    max_value = max(1e-6, max(map(abs, flat)))
    return [v / max_value for v in flat]


def load_labels(csv_path: str) -> List[str]:
    labels = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                labels.append(row[0].strip())
    return labels


class GestureApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.classifier = KeypointClassifier(
            model_path=os.path.join(
                CLASSIFIER_REPO, "model", "keypoint_classifier", "keypoint_classifier.tflite"
            )
        )
        self.labels = load_labels(
            os.path.join(
                CLASSIFIER_REPO, "model", "keypoint_classifier", "keypoint_classifier_label.csv"
            )
        )
        self.vol = VolumeController(step_percent=3)

        self.state = HandState.IDLE
        self.state_buffer: Deque[HandState] = deque(maxlen=3)

        self.index_points: Deque[Tuple[float, float]] = deque(maxlen=20)
        self.accum_angle = 0.0
        self.last_volume_time = 0.0

    def _label_to_state(self, label: str) -> HandState:
        text = label.lower()
        if "pointer" in text:
            return HandState.ADJUSTING
        if "open" in text:
            return HandState.NAVIGATING
        if "fist" in text or "close" in text:
            return HandState.IDLE
        return HandState.IDLE

    def _stable_state(self, candidate: HandState) -> HandState:
        self.state_buffer.append(candidate)
        if len(self.state_buffer) == 3 and len(set(self.state_buffer)) == 1:
            self.state = self.state_buffer[-1]
        return self.state

    def _update_volume_by_rotation(self) -> str:
        if len(self.index_points) < 6:
            return "Collecting motion..."

        pts = np.array(self.index_points, dtype=np.float64)
        center = np.mean(pts[-10:], axis=0)
        p1 = pts[-2] - center
        p2 = pts[-1] - center
        a1 = math.atan2(-p1[1], p1[0])
        a2 = math.atan2(-p2[1], p2[0])
        delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        self.accum_angle += delta

        now = time.time()
        if now - self.last_volume_time < 0.12:
            return "Adjusting..."

        trigger = math.radians(28)
        if self.accum_angle <= -trigger:
            vol = self.vol.increase()
            self.accum_angle = 0.0
            self.last_volume_time = now
            return f"CW -> Volume UP ({vol if vol is not None else 'n/a'}%)"
        if self.accum_angle >= trigger:
            vol = self.vol.decrease()
            self.accum_angle = 0.0
            self.last_volume_time = now
            return f"CCW -> Volume DOWN ({vol if vol is not None else 'n/a'}%)"
        return "Adjusting..."

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        action_text = "None"
        pred_label = "N/A"

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand_lm = result.multi_hand_landmarks[0]
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                    landmark_list = calc_landmark_list(frame, hand_lm)
                    pre = pre_process_landmark(landmark_list)
                    class_id = self.classifier(pre)
                    pred_label = self.labels[class_id] if class_id < len(self.labels) else "Unknown"

                    candidate_state = self._label_to_state(pred_label)
                    stable = self._stable_state(candidate_state)

                    idx_tip = hand_lm.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_px, y_px = idx_tip.x * w, idx_tip.y * h
                    cv2.circle(frame, (int(x_px), int(y_px)), 6, (0, 255, 255), -1)

                    if stable == HandState.ADJUSTING:
                        self.index_points.append((x_px, y_px))
                        action_text = self._update_volume_by_rotation()
                    else:
                        self.index_points.clear()
                        self.accum_angle = 0.0
                        action_text = "Navigation ready" if stable == HandState.NAVIGATING else "Idle"
                else:
                    self.state = HandState.IDLE
                    self.state_buffer.clear()
                    self.index_points.clear()
                    self.accum_angle = 0.0
                    pred_label = "No hand"
                    action_text = "Idle"

                cv2.putText(
                    frame,
                    f"Classifier: {pred_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"State: {self.state.name}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Action: {action_text}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (50, 220, 50),
                    2,
                )
                cv2.putText(
                    frame,
                    "q = quit",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    2,
                )

                cv2.imshow("GestureApp - Classifier Controlled Volume", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    app = GestureApp()
    app.run()


if __name__ == "__main__":
    main()
