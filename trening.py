import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

print("=" * 50)
print("   AI Sign Language - Improved Training")
print("=" * 50)

if not os.path.exists('podaci.csv'):
    print("Error: podaci.csv not found!")
    exit()

data = pd.read_csv('podaci.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(f"\nDataset: {len(X)} samples, {len(np.unique(y))} unique signs")
print(f"Signs: {', '.join(np.unique(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[1/4] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print(f"      RF Accuracy: {rf_score * 100:.2f}%")

print("\n[2/4] Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)
print(f"      GB Accuracy: {gb_score * 100:.2f}%")

print("\n[3/4] Creating Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)
ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(f"      Ensemble Accuracy: {ensemble_score * 100:.2f}%")

print("\n[4/4] Cross-Validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
print(f"      CV Score: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

model = ensemble
model.fit(X_train, y_train)

print("\n" + "=" * 50)
print("   Classification Report")
print("=" * 50)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\nSaving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as model.pkl")

with open('labels.pkl', 'wb') as f:
    pickle.dump(list(np.unique(y)), f)
print("Labels saved as labels.pkl")

print("\n[Training Complete!]")
