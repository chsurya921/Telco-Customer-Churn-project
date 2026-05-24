
# 📡 Telco Customer Churn Prediction System

A complete, production-ready machine learning system that predicts whether a telecom customer will cancel their subscription — and serves those predictions through a validated, monitored REST API.

---

## 🎯 What This Project Does

This project predicts whether a telecom customer will cancel their subscription — known as **customer churn** — using machine learning. It goes beyond just building a model: it packages the entire prediction system into a deployable API with data validation and live monitoring.

---

## 📊 The Problem

Telecom companies lose significant revenue when customers leave for competitors. The business challenge is identifying **which customers are at risk of leaving before they actually do**, so the company can take action — offering discounts, better plans, or targeted support — to retain them.

---

## 🔢 The Data

The dataset contains **7,043 real telecom customer records**, each with 19 features describing:

| Category | Features |
|----------|----------|
| **Demographics** | Gender, age group (senior citizen), partner, dependents |
| **Services used** | Phone service, internet type (DSL/Fiber), streaming, online security, device protection |
| **Account details** | Tenure, contract type, payment method, monthly charges, total charges |
| **Target** | Whether the customer churned — `Yes` / `No` |

---

## 🧠 The Model

The data was cleaned and prepared — handling blank values in `TotalCharges`, encoding text categories into numbers, and scaling numeric features so they're on the same scale.

Two models were trained and compared:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | ~80% | Baseline — simple and interpretable |
| **Random Forest** | **~85%** | **Final model** — better at capturing complex patterns |

Random Forest was chosen as the production model because it handles non-linear relationships better — for example, how contract type and monthly charges *together* influence churn.

### 📈 Key Findings from the Data

- Customers on **month-to-month contracts** churn far more than those on yearly contracts
- Customers with **higher monthly charges** are more likely to leave
- **Newer customers** (low tenure) have a much higher churn risk than long-term customers

---

## 🌐 The API

The trained model is served via a **REST API built with FastAPI**. Any application or system can send a customer's details as a JSON request and receive back:

- A **prediction** — will this customer churn or not (`0` or `1`)
- A **probability** — how confident the model is (`0.0` to `1.0`)
- A **risk level** — `Low`, `Medium`, or `High` based on the probability

It supports both single customer predictions and bulk batch predictions. Interactive documentation is auto-generated at `/docs`.

### Example Request
```json
POST /predict
{
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
}
```

### Example Response

<img width="1234" height="215" alt="Screenshot 2026-05-25 at 12 50 18 AM" src="https://github.com/user-attachments/assets/9620e3f8-e231-431d-ae31-c24b0a834700" />
---

## 🛡️ Data Validation

Before any data reaches the model, every incoming request passes through a **validation pipeline** that runs five layers of checks:

| Check | What it catches |
|-------|----------------|
| **Presence** | Missing or null required fields |
| **Type** | Wrong data types (e.g. tenure sent as a string) |
| **Category** | Invalid options (e.g. Contract = "Weekly" is not allowed) |
| **Range** | Unrealistic values (e.g. negative tenure, SeniorCitizen = 5) |
| **Drift** | Values unusually far from what the model was trained on |

This prevents silent wrong predictions caused by bad or unexpected input data. Errors block the prediction entirely; drift warnings flag unusual values without blocking.

---

## 📡 Monitoring

The deployed API tracks everything happening in production:

| Metric | Description |
|--------|-------------|
| **Prediction volume** | How many customers are being scored per hour |
| **Churn rate** | Percentage predicted as churners — a sudden spike may indicate a data problem |
| **Response latency** | How fast the model responds, tracked at avg, p95, and p99 |
| **Error rate** | How many requests are failing |
| **Risk distribution** | Breakdown of Low / Medium / High risk customers |

### Monitoring Interfaces

- **`/metrics`** — Prometheus-compatible endpoint, plugs into Grafana dashboards
- **`/dashboard`** — Live HTML dashboard, auto-refreshes every 10 seconds
- **`/logs/summary`** — Aggregated stats from every prediction ever logged

---

## 🏗️ System Architecture

```
Telco CSV Dataset
       │
       ▼
Data Cleaning & Feature Engineering
       │
       ▼
Model Training (Random Forest — ~85% accuracy)
       │
       ▼
Validation Pipeline (catches bad input before prediction)
       │
       ▼
REST API (serves predictions over HTTP)
       │
       ▼
Monitoring (tracks behaviour in production)
```

---

## 📂 Project Structure

```
Telco-Customer-Churn/
├── data/
│   └── Telco-Customer-Churn.csv     # Dataset
├── app/
│   └── main.py                      # FastAPI inference service
├── validation/
│   └── validate.py                  # Feature validation pipeline
├── monitoring/
│   └── monitored_api.py             # Monitored API with metrics & dashboard
├── train.py                         # Model training script
├── evaluate.py                      # Model evaluation script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
# Generates model.pkl and scaler.pkl
```

### 3. Run the API
```bash
# Basic inference service
uvicorn app.main:app --reload --port 8000

# Monitored endpoint (recommended)
uvicorn monitoring.monitored_api:app --reload --port 8000
```

### 4. Run with Docker
```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

### 5. Open in browser
| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | Interactive API documentation |
| `http://localhost:8000/dashboard` | Live monitoring dashboard |
| `http://localhost:8000/metrics` | Prometheus metrics |

---

## 📊 Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|--------------------:|-------------:|
| Accuracy | ~80% | ~85% |
| Precision | ~78% | ~83% |
| Recall | ~76% | ~81% |
| F1 Score | ~77% | ~82% |

---

## 🔗 Related Links

- 📓 Training Notebook: [Google Colab](https://colab.research.google.com/drive/1jJgX1jpG_vXmGAko8xjO7xB0AlnXIcY_?usp=sharing)
- 📦 Dataset: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

> The end result is not just a machine learning model — it's a complete, production-ready churn prediction system that can be integrated into any telecom business workflow.
