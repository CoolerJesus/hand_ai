# 🖐️ AI Sign Language Translator

This project is a machine learning-based tool that tracks hand gestures via webcam and translates them into text and spoken language. [cite_start]It uses **MediaPipe** for hand landmark extraction and a **Random Forest Classifier** for recognition.

## 🚀 Features
* [cite_start]**Real-time Tracking**: Captures 21 hand landmarks in 3D space.
* [cite_start]**Custom Dataset**: Record your own signs (e.g., "Hvala", "Bok") directly into a CSV file.
* [cite_start]**Voice Feedback**: Uses `pyttsx3` to speak recognized signs aloud.
* [cite_start]**Probability Filter**: High-confidence detection (80%+) ensures accuracy before speaking.

---

## 🛠️ Installation

1. **Install Python** (3.8+ recommended).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt