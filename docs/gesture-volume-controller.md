# AirGesture — Volume Controller Documentation

## Overview

`gesture_volume.py` is a real-time hand gesture controller built with a robust state machine.
It uses your webcam, MediaPipe Hands, and OpenCV to detect hand poses and map them to
system volume changes and navigation swipes.

A secondary classifier-driven app (`gesture_volume_classifier_app.py`) also exists but is
**not the primary script** — `gesture_volume.py` is the recommended version.

---

## Environment

| Requirement | Value |
|---|---|
| Python | 3.11.x |
| MediaPipe | 0.10.14 |
| TensorFlow | 2.13.1 |
| Protobuf | 4.25.3 |
| OpenCV | opencv-python (any recent) |
| Windows volume | pycaw + comtypes |

```powershell
cd "D:\Studies\Projects\AirGuesture"
.\.venv311\Scripts\Activate.ps1
python gesture_volume.py
```

---

## Architecture — State Machine

### States

| State | Trigger | Available actions |
|---|---|---|
| `IDLE` | Fist or rest | None |
| `ADJUSTING` | Index-only pointer pose (strict) | Rotate for volume UP/DOWN |
| `NAVIGATING` | Open palm (4+ fingers) | Swipe LEFT/RIGHT |

### State transition rules

- **IDLE → ADJUSTING**: requires **3 consecutive frames** of strict index-only pose (~100 ms hold). Prevents accidental entry from borderline frames.
- **ADJUSTING → IDLE/NAVIGATING**: requires **10 consecutive frames** of the new proposed state (~333 ms hold). Makes ADJUSTING very sticky during rotation.
- **Any → IDLE on hand-lost**: ADJUSTING holds for **5 frames** after the hand leaves the frame, then drops to IDLE.

---

## Gesture Engine

### Finger Detection — `get_finger_up_map_binary()`

Each finger is classified as up/down by comparing:
- **Tip-to-wrist** distance vs **PIP/IP-to-wrist** distance.
- A middle-finger guard prevents false positives.
- **Thumb is excluded from finger count** (unreliable in pointer pose).

### Strict Index-Only Pose — `is_index_only_pose()`

Beyond basic finger detection, the pointer pose requires:
1. Index tip clearly past its PIP joint (×1.05).
2. Middle, ring, and pinky tips within **1.55× their MCP-to-wrist** distance — i.e. bent past the knuckle line, not necessarily in a full fist.

### Continuous Rotation Tracker — `ContinuousRotationTracker`

Replaces legacy batch CW/CCW trajectory classification.

- Tracks the **signed angle** of the index tip around the wrist landmark **every frame**.
- Fires a discrete volume step every **30°** of rotation.
- In screen coordinates: **CW = positive angle = Volume UP**, **CCW = Volume DOWN**.
- `min_radius_px = 22`: angle is ignored when the tip is too close to the wrist.
- Tracker resets when the hand leaves ADJUSTING state.

### Swipe Detection — `classify_trajectory()`

Swipes are classified on the nav-point buffer every 200 ms in NAVIGATING state.

A swipe is accepted when **all** of:
| Condition | Threshold |
|---|---|
| Linearity score | ≥ 0.60 |
| Duration | ≤ 0.40 s |
| Velocity | ≥ 280 px/s |
| Displacement | ≥ 55 px |
| Horizontal dominance | `abs(dx) > abs(dy) × 1.4` |
| Directional consistency | ≥ 72% of steps in dominant direction |

---

## Landmark Smoothing

- **`WeightedLandmarkSmoother`** — linearly weighted 5-frame average over all 21 MediaPipe landmarks. Newer frames weighted more heavily.
- **`MovingAverage2D`** — simple 5-frame average applied specifically to the index fingertip for trajectory recording.

---

## Threshold Configuration Reference

All tunable parameters are in `threshold_config` at the top of `gesture_volume.py`.

| Key | Value | Description |
|---|---|---|
| `linearity_threshold` | 0.60 | Min linearity for swipe |
| `swipe_max_duration_s` | 0.40 | Max swipe duration |
| `swipe_min_velocity_px_s` | 280.0 | Min swipe velocity |
| `swipe_min_displacement_px` | 55.0 | Min swipe travel distance |
| `swipe_direction_consistency` | 0.72 | Min step consistency for swipe |
| `rotation_step_deg` | 30.0 | Degrees per volume step |
| `rotation_min_radius_px` | 22.0 | Min tip-to-wrist distance for rotation |
| `index_fold_strict_scale` | 1.55 | Fold threshold for non-index fingers |
| `deadzone_px` | 5.6 | Min movement to record a trajectory point |
| `state_debounce_frames` | 10 | Frames to confirm ADJUSTING exit |
| `adjust_enter_frames` | 3 | Frames to confirm ADJUSTING entry |
| `adjust_lost_hold_frames` | 5 | Frames to hold ADJUSTING after hand lost |

---

## Volume Control (Windows)

Uses `pycaw` → `IAudioEndpointVolume` COM interface.
Each step changes volume by **4%** (configurable via `VolumeController(step_percent=4)`).
macOS is also supported via `osascript`.

---

## Running

```powershell
cd "D:\Studies\Projects\AirGuesture"
.\.venv311\Scripts\Activate.ps1
python gesture_volume.py
```

Press **`q`** to quit.
