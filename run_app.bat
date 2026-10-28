@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please follow README.md to create .venv and install dependencies first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m digit_recognition.app
if errorlevel 1 pause
