import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import os
import pandas as pd
from datetime import datetime

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Prevoditelj Znakovnog Jezika - Pro")
        self.root.geometry("600x650")
        self.root.configure(bg="#1a1a2e")
        
        self.dark_mode = True
        self.colors = self.get_colors()
        
        self.create_widgets()
        self.update_stats()

    def get_colors(self):
        if self.dark_mode:
            return {
                'bg': '#1a1a2e', 'bg2': '#16213e', 'bg3': '#0f3460',
                'fg': '#eaeaea', 'accent': '#00d9ff', 'success': '#00ff88',
                'warning': '#ffaa00', 'danger': '#ff4757'
            }
        else:
            return {
                'bg': '#f5f5f5', 'bg2': '#ffffff', 'bg3': '#e0e0e0',
                'fg': '#333333', 'accent': '#0077aa', 'success': '#00aa55',
                'warning': '#cc8800', 'danger': '#cc3344'
            }

    def create_widgets(self):
        self.root.configure(bg=self.colors['bg'])
        
        title = tk.Label(self.root, text="AI Prevoditelj Znakovnog Jezika",
                        font=("Helvetica", 22, "bold"), fg=self.colors['accent'], 
                        bg=self.colors['bg'], pady=15)
        title.pack()

        stats_frame = tk.Frame(self.root, bg=self.colors['bg2'], padx=20, pady=15)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        self.stats_label = tk.Label(stats_frame, text="", font=("Helvetica", 10),
                                    fg=self.colors['fg'], bg=self.colors['bg2'])
        self.stats_label.pack()

        btn_frame = tk.Frame(self.root, bg=self.colors['bg'])
        btn_frame.pack(pady=20)

        self.btn_style = {
            "font": ("Helvetica", 12, "bold"), "width": 30, "height": 2,
            "bd": 0, "cursor": "hand2", "relief": "flat"
        }

        self.btn_collect = tk.Button(btn_frame, text="1. PRIKUPI NOVI ZNAK",
                                      bg=self.colors['accent'], fg="white",
                                      command=self.run_collect, **self.btn_style)
        self.btn_collect.pack(pady=8)

        self.btn_batch = tk.Button(btn_frame, text="1b. BATCH PRIKUPLJANJE",
                                     bg=self.colors['bg3'], fg=self.colors['fg'],
                                     command=self.run_batch_collect, **self.btn_style)
        self.btn_batch.pack(pady=8)

        self.btn_train = tk.Button(btn_frame, text="2. TRENIRAJ AI (Unaprijeđeno)",
                                     bg=self.colors['warning'], fg="white",
                                     command=self.run_train, **self.btn_style)
        self.btn_train.pack(pady=8)

        self.btn_detect = tk.Button(btn_frame, text="3. POKRENI PREVODITELJ",
                                     bg=self.colors['success'], fg="white",
                                     command=self.run_detect, **self.btn_style)
        self.btn_detect.pack(pady=8)

        self.btn_quiz = tk.Button(btn_frame, text="4. KVIZ - Testiraj Znanje",
                                   bg=self.colors['danger'], fg="white",
                                   command=self.run_quiz, **self.btn_style)
        self.btn_quiz.pack(pady=8)

        control_frame = tk.Frame(self.root, bg=self.colors['bg'])
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="🌙 Dark Mode", bg=self.colors['bg3'],
                  command=self.toggle_theme, **self.btn_style).pack(side="left", padx=5)
        
        tk.Button(control_frame, text="📊 Statistika", bg=self.colors['bg3'],
                  command=self.show_statistics, **self.btn_style).pack(side="left", padx=5)

        self.footer = tk.Label(self.root, text="Status: Spreman", 
                               fg=self.colors['fg'], bg=self.colors['bg'], pady=15)
        self.footer.pack(side="bottom")

    def update_stats(self):
        try:
            if os.path.exists('podaci.csv'):
                df = pd.read_csv('podaci.csv', header=None)
                samples = len(df)
                signs = df.iloc[:, -1].nunique()
                sign_counts = df.iloc[:, -1].value_counts().to_dict()
                
                self.stats_label.config(
                    text=f"📊 Uzoraka: {samples} | Znakova: {signs} | "
                         f"Prosjek: {samples//max(signs,1)} uzoraka/znak"
                )
        except:
            self.stats_label.config(text="📊 Nema podataka - prikupite prvo!")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.colors = self.get_colors()
        self.root.configure(bg=self.colors['bg'])
        for widget in self.root.winfo_children():
            try:
                widget.configure(bg=self.colors['bg'])
            except:
                pass
        self.create_widgets()

    def run_collect(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Novi znak", "Unesite ime znaka (npr. 'Pozdrav'):")
        if not name:
            return
        
        with open('temp_sign.txt', 'w') as f:
            f.write(name)
        
        if os.path.exists("prikupljanje.py"):
            subprocess.Popen(["python", "prikupljanje.py"])
            self.footer.config(text=f"Status: Snimanje '{name}'...")
        else:
            messagebox.showerror("Greška", "prikupljanje.py nije pronađeno!")

    def run_batch_collect(self):
        if os.path.exists("batch_collect.py"):
            subprocess.Popen(["python", "batch_collect.py"])
            self.footer.config(text="Status: Batch prikupljanje...")
        else:
            messagebox.showinfo("Info", "Batch skripta ne postoji. Koristite '1. PRIKUPI NOVI ZNAK'")
            self.run_collect()

    def run_train(self):
        if not os.path.exists("podaci.csv"):
            messagebox.showwarning("Upozorenje", "Prvo prikupite podatke!")
            return
            
        if os.path.exists("trening.py"):
            self.footer.config(text="Status: Treniranje modela... Molimo pričekajte.")
            self.root.update()
            
            process = subprocess.run(["python", "trening.py"], 
                                     capture_output=True, text=True)
            
            result = "Trening završen!\n\n" + process.stdout[-500:] if process.stdout else "Trening dovršen!"
            messagebox.showinfo("Trening Završen", result)
            
            self.footer.config(text="Status: Model spreman!")
            self.update_stats()
        else:
            messagebox.showerror("Greška", "trening.py nije pronađen!")

    def run_detect(self):
        if not os.path.exists("model.pkl"):
            messagebox.showwarning("Upozorenje", "Prvo morate istrenirati model (Korak 2)!")
            return
            
        if os.path.exists("detekcija.py"):
            subprocess.Popen(["python", "detekcija.py"])
            self.footer.config(text="Status: Prevoditelj pokrenut ✓")
        else:
            messagebox.showerror("Greška", "detekcija.py nije pronađen!")

    def run_quiz(self):
        if not os.path.exists("model.pkl"):
            messagebox.showwarning("Upozorenje", "Trebate trenirani model za kviz!")
            return
            
        if os.path.exists("quiz.py"):
            subprocess.Popen(["python", "quiz.py"])
            self.footer.config(text="Status: Kviz pokrenut!")
        else:
            messagebox.showinfo("Info", "Kreiram quiz...")
            self.create_quiz()

    def create_quiz(self):
        quiz_code = '''import cv2
import mediapipe as mp
import pickle
import numpy as np
import random
import time
import pandas as pd

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    labels = pd.read_csv('podaci.csv', header=None).iloc[:, -1].unique()
    
    target_sign = random.choice(labels)
    score = 0
    attempts = 0
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    print(f"=== KVIZ: Pokaži mi znak za '{target_sign}' ===")
    
    while attempts < 30:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        current_pred = "..."
        all_landmarks = np.zeros(63)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                wrist = hand_lms.landmark[0]
                for j, lm in enumerate(hand_lms.landmark):
                    all_landmarks[j*3:(j+1)*3] = [lm.x-wrist.x, lm.y-wrist.y, lm.z]
                
                current_pred = model.predict([all_landmarks])[0]
                prob = np.max(model.predict_proba([all_landmarks]))
                
                if current_pred == target_sign and prob > 0.85:
                    score += 1
                    cv2.putText(frame, "TOČNO! +1 bod", (150, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow("Kviz", frame)
                    cv2.waitKey(1500)
                    target_sign = random.choice(labels)
                    attempts = 0
        
        cv2.putText(frame, f"Cilj: {target_sign}", (10, 50), 1, 2, (255, 255, 255), 2)
        cv2.putText(frame, f"Prepoznato: {current_pred}", (10, 90), 1, 2, (0, 255, 255), 2)
        cv2.putText(frame, f"Rezultat: {score}", (10, 130), 1, 2, (0, 255, 0), 2)
        cv2.imshow("Kviz", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        attempts += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\\n=== Kraj kviza! Rezultat: {score} bodova ===")
except Exception as e:
    print(f"Greška: {e}")
'''
        with open('quiz.py', 'w') as f:
            f.write(quiz_code)
        self.run_quiz()

    def show_statistics(self):
        try:
            if not os.path.exists('podaci.csv'):
                messagebox.showinfo("Statistika", "Nema dostupnih podataka!")
                return
            
            df = pd.read_csv('podaci.csv', header=None)
            signs = df.iloc[:, -1].value_counts()
            
            stats = "📊 STATISTIKA SKUPA PODATAKA\n" + "="*35 + "\n\n"
            stats += f"Ukupno uzoraka: {len(df)}\n"
            stats += f"Broj znakova: {len(signs)}\n\n"
            stats += "Uzoraka po znaku:\n"
            
            for sign, count in signs.items():
                bar = "█" * min(count // 10, 20)
                stats += f"  {sign}: {count} {bar}\n"
            
            messagebox.showinfo("Statistika", stats)
        except Exception as e:
            messagebox.showerror("Greška", f"Nije moguće učitati podatke: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
