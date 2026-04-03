@echo off
title AI Sign Language Translator
color 0A
echo.
echo  Starting AI Sign Language Translator...
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
) else (
    echo WARNING: Virtual environment not found. Using system Python.
)

echo.
echo  Open in browser: http://localhost:5000
echo  Press Ctrl+C to stop
echo.
python app.py
