import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import pyttsx3
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier

class SignLanguagePro:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sign Language Pro")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # TTS Setup
        self.engine = pyttsx3.init()
        self.last_spoken = ""

        # UI Layout
        self.create_widgets()
        
        self.cap = None
        self.is_detecting = False
        self.model = self.load_model()

    def create_widgets(self):
        # Left Panel (Controls)
        self.sidebar = tk.Frame(self.root, bg="#34495e", width=250)
        self.sidebar.pack(side="left", fill="y")

        tk.Label(self.sidebar, text="CONTROL PANEL", font=("Arial", 14, "bold"), bg="#34495e", fg="white").pack(pady=20)

        self.btn_collect = ttk.Button(self.sidebar, text="1. Record Sign", command=self.snimi_znak)
        self.btn_collect.pack(pady=10, padx=20, fill="x")

        self.btn_train = ttk.Button(self.sidebar, text="2. Train AI", command=self.treniraj)
        self.btn_train.pack(pady=10, padx=20, fill="x")

        self.btn_toggle = tk.Button(self.sidebar, text="3. START DETECTION", bg="#27ae60", fg="white", 
                                   command=self.toggle_detection, font=("Arial", 10, "bold"))
        self.btn_toggle.pack(pady=10, padx=20, fill="x")

        # Right Panel (Video)
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.status_label = tk.Label(self.root, text="Status: Ready", bg="#2c3e50", fg="white")
        self.status_label.pack(side="bottom", fill="x")

    def normalize_landmarks(self, landmarks):
        """Converts absolute landmarks to relative coordinates based on the wrist."""
        base_x, base_y = landmarks[0].x, landmarks[0].y
        normalized = []
        for lm in landmarks:
            normalized.extend([lm.x - base_x, lm.y - base_y, lm.z])
        return normalized

    def load_model(self):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def toggle_detection(self):
        if not self.is_detecting:
            if not self.model:
                messagebox.showerror("Error", "No model found. Please train the AI first.")
                return
            self.cap = cv2.VideoCapture(0)
            self.is_detecting = True
            self.btn_toggle.config(text="STOP DETECTION", bg="#e74c3c")
            self.update_frame()
        else:
            self.is_detecting = False
            self.btn_toggle.config(text="START DETECTION", bg="#27ae60")
            if self.cap: self.cap.release()
            self.video_label.config(image="")

    def update_frame(self):
        if self.is_detecting:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                        
                        coords = self.normalize_landmarks(hand_lms.landmark)
                        prediction = self.model.predict([coords])[0]
                        prob = np.max(self.model.predict_proba([coords]))

                        if prob > 0.85:
                            cv2.putText(frame, f"{prediction} ({prob:.2f})", (10, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if prediction != self.last_spoken:
                                self.engine.say(prediction)
                                self.engine.runAndWait()
                                self.last_spoken = prediction

                # Convert to Tkinter image
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            
            self.root.after(10, self.update_frame)

    def snimi_znak(self):
        name = simpledialog.askstring("Record", "What is the name of this sign?")
        if not name: return
        
        cap = cv2.VideoCapture(0)
        count = 0
        messagebox.showinfo("Instructions", "Hold the sign. We will record 100 samples automatically.")
        
        while count < 100:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    coords = self.normalize_landmarks(hand_lms.landmark)
                    coords.append(name)
                    pd.DataFrame([coords]).to_csv('podaci.csv', mode='a', index=False, header=False)
                    count += 1
                    cv2.putText(frame, f"Recording: {count}%", (50, 50), 1, 2, (0,255,0), 2)
            
            cv2.imshow("Recording...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", f"Recorded 100 samples for '{name}'")

    def treniraj(self):
        try:
            df = pd.read_csv('podaci.csv', header=None)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            self.model = RandomForestClassifier(n_estimators=100).fit(X, y)
            with open('model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            messagebox.showinfo("Success", "Model trained and saved!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguagePro(root)
    root.mainloop()
