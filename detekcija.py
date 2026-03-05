import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
from collections import Counter

# --- CONFIGURATION ---
STABILITY_THRESHOLD = 10  # Number of frames a sign must be stable
CONFIDENCE_MIN = 0.85     # AI certainty required

engine = pyttsx3.init()
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prediction_buffer = []
current_sentence = []

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Normalization Logic [New Feature]
            wrist = hand_lms.landmark[0]
            coords = []
            for lm in hand_lms.landmark:
                coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            # Prediction
            pred = model.predict([coords])[0]
            prob = np.max(model.predict_proba([coords]))

            if prob > CONFIDENCE_MIN:
                prediction_buffer.append(pred)
                if len(prediction_buffer) > STABILITY_THRESHOLD:
                    prediction_buffer.pop(0)
                
                # Check for stability (Temporal Smoothing)
                most_common = Counter(prediction_buffer).most_common(1)[0]
                if most_common[1] == STABILITY_THRESHOLD:
                    stable_sign = most_common[0]
                    
                    # Sentence Builder Logic
                    if not current_sentence or stable_sign != current_sentence[-1]:
                        current_sentence.append(stable_sign)
                        engine.say(stable_sign)
                        engine.runAndWait()

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # UI Overlay
    sentence_text = " ".join(current_sentence[-5:]) # Show last 5 words
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence_text}", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Translator Pro", frame)
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): current_sentence = [] # Clear sentence

cap.release()
cv2.destroyAllWindows()