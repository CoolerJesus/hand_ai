import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# --- CONFIG ---
ZNAK = "Kuca" # Example: House
SAMPLES_NEEDED = 100

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data = []

print(f"Recording '{ZNAK}'. Press 'S' to start burst capture.")

while len(data) < SAMPLES_NEEDED:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Initialize empty landmarks for 2 hands (42 landmarks total, 126 values)
    all_landmarks = np.zeros(126) 

    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i > 1: break # Only take first 2 hands
            
            # Normalize relative to wrist
            wrist = hand_lms.landmark[0]
            for j, lm in enumerate(hand_lms.landmark):
                index = (i * 63) + (j * 3)
                all_landmarks[index] = lm.x - wrist.x
                all_landmarks[index+1] = lm.y - wrist.y
                all_landmarks[index+2] = lm.z - wrist.z
            
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Samples: {len(data)}/{SAMPLES_NEEDED}", (10, 50), 1, 2, (0,255,0), 2)
    cv2.imshow("Data Collection", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'): # Start recording samples
        row = list(all_landmarks) + [ZNAK]
        data.append(row)
    if key == ord('q'): break

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('podaci.csv', mode='a', index=False, header=False)
cap.release()
cv2.destroyAllWindows()