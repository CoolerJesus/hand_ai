@echo off
title AI Sign Language Translator - Setup
color 0A
echo.
echo  ========================================
echo     AI Sign Language Translator
echo        Easy Install Script
echo  ========================================
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo OK: Python found

echo.
echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo OK: Virtual environment created
) else (
    echo OK: Virtual environment exists
)

echo.
echo [3/4] Installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK: Dependencies installed

echo.
echo [4/4] Testing webcam access...
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK: Webcam' if cap.isOpened() else 'WARNING: No webcam'); cap.release()"

echo.
echo ========================================
echo     Installation Complete!
echo ========================================
echo.
echo To run the app:
echo.
echo   1. Activate:    venv\Scripts\activate
echo   2. Run app:    python app.py
echo   3. Open:       http://localhost:5000
echo.
echo Or use run.bat for quick start
echo.
pause
