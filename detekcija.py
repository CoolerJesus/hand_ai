import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
from collections import Counter, deque
import os
from datetime import datetime

CONFIDENCE_THRESHOLD = 0.75
STABILITY_FRAMES = 8
GESTURE_TIMEOUT = 30
DARK_MODE = True

engine = pyttsx3.init()
engine.setProperty('rate', 140)

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Error: model.pkl not found. Train the AI first!")
        exit()

def load_labels():
    try:
        with open('labels.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return []

model = load_model()
labels = load_labels()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prediction_buffer = deque(maxlen=STABILITY_FRAMES)
current_sentence = []
history = []
last_spoken = ""
last_sign_time = 0
paused = False
show_confidence_bar = True

BG_COLOR = (30, 30, 30) if DARK_MODE else (240, 240, 240)
TEXT_COLOR = (255, 255, 255) if DARK_MODE else (0, 0, 0)
ACCENT_COLOR = (100, 200, 100) if DARK_MODE else (0, 150, 0)

print("\n" + "=" * 50)
print("  AI Sign Language Translator - Enhanced")
print("=" * 50)
print("  Controls:")
print("    SPACE  - Pause/Resume detection")
print("    C      - Clear sentence")
print("    Z      - Undo last word")
print("    S      - Save sentence to file")
print("    H      - Show/Hide confidence bar")
print("    Q      - Quit")
print("=" * 50 + "\n")

def draw_ui(frame, prediction, confidence, sentence):
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (0, h-80), (w, h), (20, 20, 20), -1)
    
    sentence_str = " ".join(sentence[-8:]) if sentence else "..."
    cv2.putText(frame, f"Sentence: {sentence_str}", (20, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if show_confidence_bar and confidence > 0:
        bar_width = int(w * 0.4)
        bar_x = w - bar_width - 20
        bar_y = h - 50
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (60, 60, 60), -1)
        fill_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.5 else (0, 100, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 15), color, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (bar_x + bar_width + 10, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if paused:
        cv2.rectangle(frame, (w//2 - 150, h//2 - 40), (w//2 + 150, h//2 + 40), (0, 0, 0), -1)
        cv2.putText(frame, "PAUSED - Press SPACE", (w//2 - 130, h//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def normalize_landmarks(landmarks, wrist):
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z])
    return normalized

def speak(text):
    engine.say(text)
    engine.runAndWait()

def save_sentence(sentence):
    if not sentence:
        return
    filename = f"sentences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(" ".join(sentence))
    print(f"Sentence saved to: {filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    all_landmarks = np.zeros(126)
    current_prediction = "..."
    current_confidence = 0

    if results.multi_hand_landmarks and not paused:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1:
                break
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   landmark_drawing_spec=mp_draw.DrawingSpec(
                                       color=(0, 200, 100) if i == 0 else (200, 100, 0),
                                       thickness=2, circle_radius=3))
            
            wrist = hand_landmarks.landmark[0]
            coords = normalize_landmarks(hand_landmarks.landmark, wrist)
            all_landmarks[i*63:(i+1)*63] = coords

        try:
            prediction = model.predict([all_landmarks])[0]
            probabilities = model.predict_proba([all_landmarks])[0]
            confidence = np.max(probabilities)
            current_prediction = prediction
            current_confidence = confidence

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(prediction)
                last_sign_time = 0

                if len(prediction_buffer) == STABILITY_FRAMES:
                    most_common = Counter(prediction_buffer).most_common(1)[0]
                    if most_common[1] >= STABILITY_FRAMES:
                        stable_sign = most_common[0]
                        if not current_sentence or stable_sign != current_sentence[-1]:
                            current_sentence.append(stable_sign)
                            history.append(stable_sign)
                            last_spoken = stable_sign
                            speak(stable_sign)
        except:
            pass

    draw_ui(frame, current_prediction, current_confidence, current_sentence)

    label_text = f"{current_prediction} ({current_confidence*100:.0f}%)"
    cv2.putText(frame, label_text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if labels:
        help_text = "Signs: " + ", ".join(labels[:5]) + ("..." if len(labels) > 5 else "")
        cv2.putText(frame, help_text, (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("AI Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('c'):
        current_sentence = []
        print("Sentence cleared")
    elif key == ord('z'):
        if current_sentence:
            removed = current_sentence.pop()
            print(f"Removed: {removed}")
    elif key == ord('s'):
        save_sentence(current_sentence)
    elif key == ord('h'):
        show_confidence_bar = not show_confidence_bar

cap.release()
cv2.destroyAllWindows()
print("\nGoodbye!")
