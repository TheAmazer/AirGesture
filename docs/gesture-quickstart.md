# AirGesture — Quickstart

## 1) Open project

```powershell
cd "D:\Studies\Projects\AirGuesture"
```

## 2) Activate environment

```powershell
.\.venv311\Scripts\Activate.ps1
```

## 3) Run the state-machine app

```powershell
python gesture_volume.py
```

Or use the launcher:

```powershell
.\run_gesture.bat
```

## 4) Hand Gesture Reference

| Pose | Fingers | State | Action |
|---|---|---|---|
| Fist / rest | 0 fingers | `IDLE` | Nothing |
| Pointer | Index only (others folded) | `ADJUSTING` | Rotate index tip CW/CCW for volume |
| Open palm | 4+ fingers extended | `NAVIGATING` | Swipe left / right |

### Volume control (ADJUSTING state)
- Hold your **index finger up** with the other fingers folded — hold the pose for ~100 ms until you see `State: ADJUSTING` on screen.
- **Rotate your wrist / index tip clockwise** → Volume UP
- **Rotate counter-clockwise** → Volume DOWN
- Each ~30° of rotation fires one volume step.

### Swipe (NAVIGATING state)
- Open your **full palm** to enter `NAVIGATING`.
- **Quick horizontal swipe** left or right registers as `Swipe Left` / `Swipe Right`.

## 5) Debug overlay (on-screen)

| Label | Meaning |
|---|---|
| `State` | Current hand state (IDLE / ADJUSTING / NAVIGATING) |
| `Action` | Last triggered action |
| `Fingers` | Count of extended fingers (thumb excluded) |
| `Linearity` | Swipe linearity score (0–1) |
| `Dir Consistency` | Swipe directional consistency score (0–1) |
| `Rot accum` | Accumulated rotation angle in degrees |
| `Rot lock` | Whether ADJUSTING state is locked |

## 6) Exit

Press `q`.
