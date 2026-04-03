#!/bin/bash

echo "========================================"
echo "  AI Sign Language Translator"
echo "     Easy Install Script"
echo "========================================"

echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Install from: https://python.org"
    exit 1
fi
echo "OK: Python found"

echo "[2/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Testing webcam..."
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK: Webcam found' if cap.isOpened() else 'WARNING: No webcam'); cap.release()"

echo ""
echo "========================================"
echo "     Installation Complete!"
echo "========================================"
echo ""
echo "To run:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Open: http://localhost:5000"
echo ""
