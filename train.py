"""
train.py — Telco Customer Churn Model Training
Trains RandomForest model and saves model.pkl + scaler.pkl
Run: python train.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.getenv("DATA_PATH", "data/Telco-Customer-Churn.csv")

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop("customerID", axis=1, inplace=True)

# ── Preprocess ───────────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── Save ──────────────────────────────────────────────────────────────────────
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Saved model.pkl and scaler.pkl")
