import cv2
import mediapipe as mp
import csv

# POSTAVKE
ZNAK = "Hvala"  # Promijeni u "Bok", "Da", "Ne", itd.
FILE_NAME = 'podaci.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(f"Snimam za: {ZNAK}. Pritisni 'S' za spremanje, 'Q' za izlaz.")

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                row = [val for lm in hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
                row.append(ZNAK)
                with open(FILE_NAME, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
                print("Spremljeno!")

    cv2.imshow("Snimanje Data-seta", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
