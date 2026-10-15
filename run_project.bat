@echo off
title SmartTraffic AI Control Center
cd /d "%~dp0"
echo ================================================================
echo   🚦 Launching SmartTraffic AI Control Center...
echo ================================================================

rem Open browser directly
start "" http://127.0.0.1:5000

rem Check for virtual environments
if exist "..\venv311\Scripts\python.exe" (
    "..\venv311\Scripts\python.exe" app.py
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" app.py
) else (
    python app.py
)

pause
