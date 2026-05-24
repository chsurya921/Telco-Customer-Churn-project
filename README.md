# 📡 Telco Customer Churn — ML Inference & Monitoring

> ScholarX ML Engineer Internship | Tasks 2, 3 & 4

---

## 🗂️ Project Structure

```
Telco-Customer-Churn/
├── data/
│   └── Telco-Customer-Churn.csv
├── app/
│   └── main.py              # Task 2: Model Inference Service (FastAPI)
├── validation/
│   └── validate.py          # Task 3: Feature Validation Pipeline
├── monitoring/
│   └── monitored_api.py     # Task 4: Monitored ML Endpoint
├── train.py                 # Model training script
├── evaluate.py              # Evaluation script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
# Saves model.pkl and scaler.pkl
```

---

## 🚀 Task 2 — Model Inference Service

A production-ready REST API built with **FastAPI** that serves churn predictions.

### Run locally
```bash
uvicorn app.main:app --reload --port 8000
```

### API Endpoints

| Method | Endpoint        | Description                  |
|--------|----------------|------------------------------|
| GET    | `/`            | Service info                 |
| GET    | `/health`      | Health check                 |
| POST   | `/predict`     | Single customer prediction   |
| POST   | `/predict/batch` | Batch predictions          |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.90,
    "TotalCharges": 1340.00
}'
```

### Example Response
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.1823,
  "risk_level": "Low",
  "message": "Customer likely to stay."
}
```

### Interactive Docs
Visit `http://localhost:8000/docs` for Swagger UI.

---

## ✅ Task 3 — Feature Validation Pipeline

A standalone validation module that catches bad data **before** it reaches the model.

### Checks performed
- **Presence** — all required fields exist and are non-null  
- **Type** — correct Python type for each field  
- **Category** — categorical values match allowed options  
- **Range** — numeric values within expected bounds  
- **Drift** — values flagged if >3 std deviations from training mean  

### Run standalone
```bash
python validation/validate.py
```

### Use in code
```python
from validation.validate import FeatureValidator

validator = FeatureValidator()

record = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 24, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "One year", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.90, "TotalCharges": 1340.00
}

report = validator.validate_record(record)
print(report.to_dict())
# {"is_valid": true, "errors": [], "warnings": [], ...}
```

### Batch validation (DataFrame)
```python
import pandas as pd
from validation.validate import FeatureValidator

df = pd.read_csv("data/Telco-Customer-Churn.csv")
validator = FeatureValidator()
valid_df, invalid_df, reports = validator.validate_dataframe(df)
print(f"{len(valid_df)} valid rows, {len(invalid_df)} invalid rows")
```

---

## 📊 Task 4 — Monitored ML Endpoint

Extends the inference API with full production monitoring:

- **Prometheus `/metrics`** — counters, gauges for requests, latency, churn rate  
- **HTML `/dashboard`** — live dashboard (auto-refreshes every 10 seconds)  
- **Prediction logging** — appends every prediction to `prediction_logs.jsonl`  
- **`/logs/summary`** — aggregated stats from log file  

### Run locally
```bash
uvicorn monitoring.monitored_api:app --reload --port 8000
```

### Monitoring Endpoints

| Endpoint         | Description                            |
|-----------------|----------------------------------------|
| `/health`        | Model status, uptime, error rate       |
| `/metrics`       | Prometheus-compatible metrics          |
| `/dashboard`     | HTML monitoring dashboard              |
| `/logs/summary`  | Summary of prediction logs             |

### Run with Docker
```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

Then open:
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`
- Metrics: `http://localhost:8000/metrics`

---

## 📈 Metrics Tracked

| Metric | Type | Description |
|--------|------|-------------|
| `telco_requests_total` | Counter | Total prediction requests |
| `telco_churn_predictions_total` | Counter | Churn predictions made |
| `telco_churn_rate` | Gauge | Current churn rate |
| `telco_errors_total` | Counter | Failed requests |
| `telco_latency_avg_ms` | Gauge | Average inference latency |
| `telco_latency_p95_ms` | Gauge | 95th percentile latency |
| `telco_risk_*_total` | Counter | Per risk-tier counts |

---

## 🔗 Related

- Training notebook: [Google Colab](https://colab.research.google.com/drive/1jJgX1jpG_vXmGAko8xjO7xB0AlnXIcY_?usp=sharing)
- Dataset: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

*Built for ScholarX ML Engineer Internship — Tasks 2, 3 & 4*
