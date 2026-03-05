```python
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
from collections import Counter

# --- SETTINGS ---
CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 10  # Frames a sign must stay the same to be counted

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load Model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found. Please train the AI first.")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Tracking Variables
prediction_buffer = []
current_sentence = []
last_spoken = ""

print("Detection started. Press 'C' to clear sentence, 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Prepare features (126 values: 2 hands * 21 landmarks * 3 coordinates)
    all_landmarks = np.zeros(126)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1: break # Only process first two hands
            
            # Draw on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalize relative to wrist (landmark 0)
            wrist = hand_landmarks.landmark[0]
            for j, lm in enumerate(hand_landmarks.landmark):
                idx = (i * 63) + (j * 3)
                all_landmarks[idx] = lm.x - wrist.x
                all_landmarks[idx+1] = lm.y - wrist.y
                all_landmarks[idx+2] = lm.z - wrist.z

        # Perform Prediction
        prediction = model.predict([all_landmarks])[0]
        probability = np.max(model.predict_proba([all_landmarks]))

        if probability > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > STABILITY_FRAMES:
                prediction_buffer.pop(0)

            # Check for stability
            most_common = Counter(prediction_buffer).most_common(1)[0]
            if most_common[1] == STABILITY_FRAMES:
                stable_sign = most_common[0]

                # Update Sentence
                if not current_sentence or stable_sign != current_sentence[-1]:
                    current_sentence.append(stable_sign)
                    engine.say(stable_sign)
                    engine.runAndWait()

            # Display prediction on hand
            cv2.putText(frame, f"{prediction} ({probability:.2f})", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # UI: Sentence Bar
    cv2.rectangle(frame, (0, h-60), (w, h), (44, 62, 80), -1)
    sentence_str = " ".join(current_sentence[-6:]) # Show last 6 words
    cv2.putText(frame, f"Sentence: {sentence_str}", (20, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("AI Sign Language Translator", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_sentence = []

cap.release()
cv2.destroyAllWindows()