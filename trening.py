import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Učitavanje podataka
data = pd.read_csv('podaci.csv', header=None)
X = data.iloc[:, :-1]  # Koordinate
y = data.iloc[:, -1]   # Labele (npr. "Hvala")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Treniranje modela
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Provjera točnosti
score = model.score(X_test, y_test)
print(f"Točnost modela: {score * 100:.2f}%")

# Spremanje modela
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model spremljen kao model.pkl")
