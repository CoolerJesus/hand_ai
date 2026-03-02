import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import pyttsx3
import tkinter as tk
from tkinter import messagebox, simpledialog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal AI Sign Language")
        self.root.geometry("300x400")
        
        # Inicijalizacija MediaPipe-a
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # GUI Elementi
        tk.Label(root, text="AI Prevoditelj", font=("Arial", 18)).pack(pady=20)
        tk.Button(root, text="1. Snimi Znak", command=self.snimi_znak, width=20).pack(pady=5)
        tk.Button(root, text="2. Treniraj Model", command=self.treniraj, width=20).pack(pady=5)
        tk.Button(root, text="3. Pokreni Kameru", command=self.pokreni_detekciju, width=20).pack(pady=5)

    def snimi_znak(self):
        znak = simpledialog.askstring("Input", "Unesi ime znaka (npr. Bok):")
        if not znak: return
        
        cap = cv2.VideoCapture(0)
        messagebox.showinfo("Upute", "Pritisnite 'S' za snimanje okvira (snimite bar 50), 'Q' za kraj.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            results = self.hands.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for lm_list in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, lm_list, self.mp_hands.HAND_CONNECTIONS)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        data = [v for lm in lm_list.landmark for v in [lm.x, lm.y, lm.z]]
                        data.append(znak)
                        pd.DataFrame([data]).to_csv('podaci.csv', mode='a', index=False, header=False)
            cv2.imshow("Snimanje...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

    def treniraj(self):
        try:
            df = pd.read_csv('podaci.csv', header=None)
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            model = RandomForestClassifier().fit(X, y)
            with open('model.pkl', 'wb') as f: pickle.dump(model, f)
            messagebox.showinfo("Uspjeh", "Model je istreniran!")
        except Exception as e: messagebox.showerror("Greška", "Snimite podatke prvo!")

    def pokreni_detekciju(self):
        try:
            with open('model.pkl', 'rb') as f: model = pickle.load(f)
            engine = pyttsx3.init()
            cap = cv2.VideoCapture(0)
            zadnji = ""
            while cap.isOpened():
                ret, frame = cap.read()
                results = self.hands.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for lm_list in results.multi_hand_landmarks:
                        coords = [v for lm in lm_list.landmark for v in [lm.x, lm.y, lm.z]]
                        pred = model.predict([coords])[0]
                        cv2.putText(frame, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        if pred != zadnji:
                            engine.say(pred); engine.runAndWait()
                            zadnji = pred
                cv2.imshow("AI Detekcija", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release(); cv2.destroyAllWindows()
        except: messagebox.showerror("Greška", "Model nije pronađen!")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIApp(root)
    root.mainloop()
