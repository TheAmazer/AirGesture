@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv311\Scripts\python.exe" (
  echo [ERROR] Missing Python venv at .venv311
  echo Please recreate it or run gesture_volume.py manually.
  pause
  exit /b 1
)

echo Starting gesture volume controller...
".venv311\Scripts\python.exe" "gesture_volume.py"

echo.
echo Gesture app exited.
pause
