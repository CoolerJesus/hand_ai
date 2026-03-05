import cv2
import mediapipe as mp
import pickle
import numpy as np
import random
import time

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get list of unique signs from your dataset
import pandas as pd
labels = pd.read_csv('podaci.csv', header=None).iloc[:, -1].unique()

target_sign = random.choice(labels)
score = 0

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

print(f"QUIZ: Show me the sign for '{target_sign}'")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    current_pred = "None"
    all_landmarks = np.zeros(126)

    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i > 1: break
            wrist = hand_lms.landmark[0]
            for j, lm in enumerate(hand_lms.landmark):
                idx = (i * 63) + (j * 3)
                all_landmarks[idx:idx+3] = [lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z]

        current_pred = model.predict([all_landmarks])[0]
        prob = np.max(model.predict_proba([all_landmarks]))

        if current_pred == target_sign and prob > 0.9:
            cv2.putText(frame, "CORRECT!", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            cv2.imshow("Quiz", frame)
            cv2.waitKey(2000) # Pause to celebrate
            target_sign = random.choice(labels) # New word

    cv2.putText(frame, f"TARGET: {target_sign}", (10, 50), 1, 2, (255, 255, 255), 2)
    cv2.putText(frame, f"DETECTED: {current_pred}", (10, 90), 1, 2, (255, 255, 0), 2)
    cv2.imshow("Quiz", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()