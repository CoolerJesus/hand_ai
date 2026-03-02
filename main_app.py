import tkinter as tk
from tkinter import messagebox
import subprocess
import os

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Prevoditelj Znakovnog Jezika")
        self.root.geometry("400x500")
        self.root.configure(bg="#2c3e50")

        # Naslov
        self.label = tk.Label(root, text="AI Sign Translator", font=("Helvetica", 20, "bold"), 
                              fg="white", bg="#2c3e50", pady=20)
        self.label.pack()

        # Opis
        self.desc = tk.Label(root, text="Korak 1: Prikupi podatke\nKorak 2: Istreniraj sustav\nKorak 3: Pokreni prevoditelj", 
                             fg="#ecf0f1", bg="#2c3e50", font=("Helvetica", 10))
        self.desc.pack(pady=10)

        # Gumbi
        self.btn_style = {"font": ("Helvetica", 12), "width": 25, "height": 2, "bd": 0, "cursor": "hand2"}

        self.btn_collect = tk.Button(root, text="1. PRIKUPI PODATKE", bg="#3498db", fg="white", 
                                     command=self.run_collect, **self.btn_style)
        self.btn_collect.pack(pady=10)

        self.btn_train = tk.Button(root, text="2. TRENIRAJ AI", bg="#e67e22", fg="white", 
                                   command=self.run_train, **self.btn_style)
        self.btn_train.pack(pady=10)

        self.btn_detect = tk.Button(root, text="3. POKRENI PREVODITELJ", bg="#27ae60", fg="white", 
                                    command=self.run_detect, **self.btn_style)
        self.btn_detect.pack(pady=10)

        # Footer
        self.footer = tk.Label(root, text="Status: Spreman", fg="#bdc3c7", bg="#2c3e50", pady=20)
        self.footer.pack(side="bottom")

    def run_collect(self):
        # Otvara skriptu za prikupljanje (provjeri imaš li datoteku prikupljanje.py)
        if os.path.exists("prikupljanje.py"):
            subprocess.Popen(["python", "prikupljanje.py"])
            self.footer.config(text="Status: Snimanje podataka...")
        else:
            messagebox.showerror("Greška", "Datoteka prikupljanje.py nije pronađena!")

    def run_train(self):
        if os.path.exists("trening.py"):
            self.footer.config(text="Status: Treniranje u tijeku... Molimo pričekajte.")
            self.root.update()
            process = subprocess.run(["python", "trening.py"], capture_output=True, text=True)
            messagebox.showinfo("Trening završen", process.stdout)
            self.footer.config(text="Status: Model spreman!")
        else:
            messagebox.showerror("Greška", "Datoteka trening.py nije pronađena!")

    def run_detect(self):
        if os.path.exists("model.pkl"):
            subprocess.Popen(["python", "detekcija.py"])
            self.footer.config(text="Status: Prevoditelj pokrenut.")
        else:
            messagebox.showwarning("Upozorenje", "Prvo moraš istrenirati model (Korak 2)!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
