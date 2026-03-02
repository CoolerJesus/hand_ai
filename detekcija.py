import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3

# Inicijalizacija govora (Text-to-Speech)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Učitavanje modela
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
zadnji_znak = ""

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Predviđanje
            coords = [val for lm in hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
            predikcija = model.predict([coords])[0]
            vjerojatnost = np.max(model.predict_proba([coords]))

            if vjerojatnost > 0.8:  # Samo ako je AI siguran više od 80%
                cv2.putText(frame, f"{predikcija} ({vjerojatnost:.2f})", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Izgovori ako se znak promijenio
                if predikcija != zadnji_znak:
                    engine.say(predikcija)
                    engine.runAndWait()
                    zadnji_znak = predikcija

    cv2.imshow("AI Prevoditelj Znakovnog Jezika", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
