# AirGesture — Troubleshooting

## 1) `AttributeError ... FieldDescriptor ... label` (MediaPipe startup)

**Cause:** Incompatible `protobuf` version with TensorFlow/MediaPipe.

**Fix:**
```powershell
cd "D:\Studies\Projects\AirGuesture"
.\.venv311\Scripts\python.exe -m pip install "tensorflow==2.13.1" "protobuf==4.25.3"
.\.venv311\Scripts\python.exe -m pip uninstall -y jax jaxlib
```

---

## 2) `mp.solutions` missing

**Cause:** Unsupported Python/MediaPipe wheel combination.

**Fix:** Run only with `.venv311` (Python 3.11) and `mediapipe==0.10.14`.

```powershell
.\.venv311\Scripts\python.exe gesture_volume.py
```

---

## 3) `python` or `py` not recognized in terminal

**Fix:** Use the direct interpreter path:

```powershell
.\.venv311\Scripts\python.exe gesture_volume.py
```

---

## 4) State constantly flickers between IDLE and ADJUSTING

**Cause:** The pointer pose (index-only) detection is borderline — finger position
is near the `index_fold_strict_scale` threshold.

**Fixes to try (in order):**

1. **Hold the pose more deliberately** — the system requires ~100 ms (3 frames) of
   steady pointer pose before entering ADJUSTING. Wobbling fingers during entry reset the counter.

2. **Improve lighting** — poor lighting degrades MediaPipe landmark accuracy, causing
   the fold-check to flicker near the threshold.

3. **Relax the strict scale** — in `threshold_config` in `gesture_volume.py`, raise:
   ```python
   "index_fold_strict_scale": 1.55,  # try 1.65 or 1.70 if still flickering
   ```

4. **Increase entry debounce** — slower but more stable:
   ```python
   "adjust_enter_frames": 3,  # try 5 for even more stability
   ```

---

## 5) Volume changes erratically (both up and down) during rotation

**Cause:** Rotation step size too small relative to hand jitter, causing the
accumulated angle to oscillate across the step threshold.

**Fix:** Increase `rotation_step_deg` in `threshold_config`:

```python
"rotation_step_deg": 30.0,  # try 35.0 or 40.0 for less sensitivity
```

Also ensure your hand/wrist is at least ~22 px from the wrist landmark on-screen
(`rotation_min_radius_px`). If the tip is too close to the wrist, the angle is
unreliable. Move your hand slightly further from the camera.

---

## 6) Swipe is not being detected

**Cause:** Swipe thresholds not met — too slow, too short, or inconsistent direction.

**Tips:**
- Make a **quick, clean horizontal** swipe — mostly left or right, not diagonal.
- The swipe must cover at least **55 px** and complete within **0.40 s**.
- Check the `Linearity` and `Dir Consistency` values on the debug overlay while swiping;
  both should approach 1.0 for a clean swipe.

**Fix (lower thresholds if needed):**
```python
"swipe_min_velocity_px_s": 280.0,      # try 200.0
"swipe_min_displacement_px": 55.0,     # try 40.0
"swipe_direction_consistency": 0.72,   # try 0.65
```

---

## 7) No volume change (Windows)

- Verify state shows `ADJUSTING` on the debug overlay.
- Check that `pycaw` and `comtypes` are installed in the venv:
  ```powershell
  .\.venv311\Scripts\pip.exe install pycaw comtypes
  ```
- Check Windows audio device permissions — the default audio endpoint must be accessible.
- Look for `[WARN] Windows volume setup failed` in the console output for specific error details.

---

## 8) Camera not opening (`RuntimeError: Could not open webcam`)

- Ensure no other application (Teams, Zoom, OBS) is exclusively holding the camera.
- Try changing the camera index in `gesture_volume.py`:
  ```python
  cap = cv2.VideoCapture(1)  # try 1, 2, etc.
  ```
