"""
Task 2: Model Inference Service
Telco Customer Churn - FastAPI REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Optional
import pickle
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability for telecom customers",
    version="1.0.0"
)

# ── Load model & scaler at startup ──────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    logger.info("Model and scaler loaded successfully.")
except FileNotFoundError:
    logger.warning("model.pkl / scaler.pkl not found. Run train.py first.")
    model, scaler = None, None


# ── Request / Response schemas ───────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str                        # "Male" / "Female"
    SeniorCitizen: int                 # 0 or 1
    Partner: str                       # "Yes" / "No"
    Dependents: str                    # "Yes" / "No"
    tenure: int                        # months
    PhoneService: str                  # "Yes" / "No"
    MultipleLines: str                 # "Yes" / "No" / "No phone service"
    InternetService: str               # "DSL" / "Fiber optic" / "No"
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str                      # "Month-to-month" / "One year" / "Two year"
    PaperlessBilling: str              # "Yes" / "No"
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    @validator("tenure")
    def tenure_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("tenure must be >= 0")
        return v

    @validator("MonthlyCharges", "TotalCharges")
    def charges_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("charges must be >= 0")
        return v


class PredictionResponse(BaseModel):
    churn_prediction: int              # 0 = stays, 1 = churns
    churn_probability: float           # probability of churning
    risk_level: str                    # Low / Medium / High
    message: str


# ── Helper: preprocess input the same way train.py does ─────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Fitted label-encoding mappings (mirrors LabelEncoder alphabetical order)
LABEL_MAPS = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup":     {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies":  {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    },
}

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def preprocess(customer: CustomerFeatures) -> np.ndarray:
    data = customer.dict()
    row = []
    for col in FEATURE_ORDER:
        val = data[col]
        if col in LABEL_MAPS:
            if val not in LABEL_MAPS[col]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid value '{val}' for field '{col}'. "
                           f"Allowed: {list(LABEL_MAPS[col].keys())}"
                )
            row.append(LABEL_MAPS[col][val])
        else:
            row.append(val)
    arr = np.array(row).reshape(1, -1)
    arr = scaler.transform(arr)
    return arr


def risk_label(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Telco Churn Prediction API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    features = preprocess(customer)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    risk = risk_label(probability)

    logger.info(f"Prediction: {prediction}, Probability: {probability:.3f}, Risk: {risk}")

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk,
        message="Customer likely to churn." if prediction == 1 else "Customer likely to stay."
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = []
    for i, customer in enumerate(customers):
        features = preprocess(customer)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        results.append({
            "index": i,
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_label(probability)
        })
    return {"total": len(results), "predictions": results}
