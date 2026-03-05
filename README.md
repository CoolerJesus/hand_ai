# 🖐️ AI Sign Language Translator (Pro Version)

This is an advanced machine learning tool that translates sign language gestures into text and speech in real-time. It uses **MediaPipe** for hand tracking and a **Random Forest** model for gesture classification.

## ✨ Key Features
* **Two-Hand Support**: Recognizes complex signs using one or both hands.
* **Coordinate Normalization**: AI understands the hand shape regardless of where you are in the camera frame.
* **Temporal Smoothing**: Eliminates flickering by requiring a sign to be stable for 10 frames before translating.
* **Sentence Builder**: Automatically strings words together to form full sentences.
* **Voice Output**: Integrated Text-to-Speech (TTS) for accessibility.
* **Interactive Quiz**: A game mode to test your knowledge of signs.

---

## 🛠️ Installation

1. **Install Python 3.8+**
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt