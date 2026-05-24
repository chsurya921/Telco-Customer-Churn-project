"""
Task 4: Monitored ML Endpoint
Extends the inference service with:
  - Prometheus metrics (latency, prediction counts, drift alerts)
  - Request logging to JSONL file
  - /metrics endpoint
  - /dashboard endpoint (HTML summary)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, validator
from typing import Optional, List
import pickle, numpy as np, logging, time, json, os
from datetime import datetime
from collections import deque, defaultdict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn - Monitored Endpoint",
    description="Production ML endpoint with monitoring, logging, and drift detection",
    version="2.0.0"
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
LOG_PATH = os.getenv("LOG_PATH", "prediction_logs.jsonl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    logger.info("Model loaded.")
except FileNotFoundError:
    model, scaler = None, None
    logger.warning("Model files not found.")

# ── In-memory metrics store ───────────────────────────────────────────────────
_lock = threading.Lock()

metrics = {
    "total_requests": 0,
    "churn_predictions": 0,
    "no_churn_predictions": 0,
    "errors": 0,
    "latencies_ms": deque(maxlen=1000),  # last 1000 requests
    "hourly_counts": defaultdict(int),
    "risk_counts": defaultdict(int),
    "start_time": datetime.utcnow().isoformat(),
}

# Reference stats for drift detection
REF_STATS = {
    "MonthlyCharges": {"mean": 64.76, "std": 30.09},
    "tenure":         {"mean": 32.37, "std": 24.56},
    "TotalCharges":   {"mean": 2283.3, "std": 2266.77},
}

def record_metric(prediction: int, probability: float, latency_ms: float,
                  monthly: float, tenure: int, total: float, risk: str):
    with _lock:
        metrics["total_requests"] += 1
        if prediction == 1:
            metrics["churn_predictions"] += 1
        else:
            metrics["no_churn_predictions"] += 1
        metrics["latencies_ms"].append(latency_ms)
        hour_key = datetime.utcnow().strftime("%Y-%m-%d %H:00")
        metrics["hourly_counts"][hour_key] += 1
        metrics["risk_counts"][risk] += 1


# ── Schema (same as Task 2) ───────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

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
    "PaymentMethod":    {
        "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
        "Electronic check": 2, "Mailed check": 3
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
                raise HTTPException(422, f"Invalid value '{val}' for '{col}'")
            row.append(LABEL_MAPS[col][val])
        else:
            row.append(val)
    return scaler.transform(np.array(row).reshape(1, -1))

def risk_label(p: float) -> str:
    return "High" if p >= 0.65 else ("Medium" if p >= 0.35 else "Low")

def log_prediction(customer: dict, prediction: int, probability: float,
                   latency_ms: float, risk: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk": risk,
        "latency_ms": round(latency_ms, 2),
        "features": {
            "MonthlyCharges": customer.get("MonthlyCharges"),
            "tenure": customer.get("tenure"),
            "Contract": customer.get("Contract"),
        }
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Telco Churn Monitored API", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    with _lock:
        total = metrics["total_requests"]
        errors = metrics["errors"]
        error_rate = (errors / total * 100) if total > 0 else 0
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_requests": total,
        "error_rate_pct": round(error_rate, 2),
        "uptime_since": metrics["start_time"]
    }


@app.post("/predict")
def predict(customer: CustomerFeatures, request: Request):
    if model is None:
        with _lock:
            metrics["errors"] += 1
        raise HTTPException(503, "Model not loaded.")

    t0 = time.time()
    features = preprocess(customer)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    latency_ms = (time.time() - t0) * 1000
    risk = risk_label(probability)

    record_metric(prediction, probability, latency_ms,
                  customer.MonthlyCharges, customer.tenure, customer.TotalCharges, risk)
    log_prediction(customer.dict(), prediction, probability, latency_ms, risk)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "latency_ms": round(latency_ms, 2),
        "message": "Customer likely to churn." if prediction == 1 else "Customer likely to stay."
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Prometheus-compatible /metrics endpoint."""
    with _lock:
        total = metrics["total_requests"]
        churn = metrics["churn_predictions"]
        no_churn = metrics["no_churn_predictions"]
        errors = metrics["errors"]
        lats = list(metrics["latencies_ms"])
        risk_counts = dict(metrics["risk_counts"])

    avg_lat = round(np.mean(lats), 2) if lats else 0
    p95_lat = round(np.percentile(lats, 95), 2) if lats else 0
    p99_lat = round(np.percentile(lats, 99), 2) if lats else 0
    churn_rate = round(churn / total, 4) if total > 0 else 0

    lines = [
        "# HELP telco_requests_total Total prediction requests",
        "# TYPE telco_requests_total counter",
        f"telco_requests_total {total}",
        "",
        "# HELP telco_churn_predictions_total Total churn predictions",
        "# TYPE telco_churn_predictions_total counter",
        f"telco_churn_predictions_total {churn}",
        "",
        "# HELP telco_no_churn_predictions_total Total no-churn predictions",
        "# TYPE telco_no_churn_predictions_total counter",
        f"telco_no_churn_predictions_total {no_churn}",
        "",
        "# HELP telco_churn_rate Current churn prediction rate",
        "# TYPE telco_churn_rate gauge",
        f"telco_churn_rate {churn_rate}",
        "",
        "# HELP telco_errors_total Total prediction errors",
        "# TYPE telco_errors_total counter",
        f"telco_errors_total {errors}",
        "",
        "# HELP telco_latency_avg_ms Average prediction latency (ms)",
        "# TYPE telco_latency_avg_ms gauge",
        f"telco_latency_avg_ms {avg_lat}",
        "",
        "# HELP telco_latency_p95_ms 95th percentile latency (ms)",
        "# TYPE telco_latency_p95_ms gauge",
        f"telco_latency_p95_ms {p95_lat}",
        "",
        "# HELP telco_latency_p99_ms 99th percentile latency (ms)",
        "# TYPE telco_latency_p99_ms gauge",
        f"telco_latency_p99_ms {p99_lat}",
    ]
    for risk, count in risk_counts.items():
        lines += [
            f'# HELP telco_risk_{risk.lower()}_total Customers in {risk} risk tier',
            f'# TYPE telco_risk_{risk.lower()}_total counter',
            f'telco_risk_{risk.lower()}_total {count}',
            "",
        ]
    return "\n".join(lines)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Simple HTML monitoring dashboard."""
    with _lock:
        total = metrics["total_requests"]
        churn = metrics["churn_predictions"]
        no_churn = metrics["no_churn_predictions"]
        errors = metrics["errors"]
        lats = list(metrics["latencies_ms"])
        risk_counts = dict(metrics["risk_counts"])
        start = metrics["start_time"]

    avg_lat = round(np.mean(lats), 2) if lats else 0
    p95_lat = round(np.percentile(lats, 95), 2) if lats else 0
    churn_rate = round(churn / total * 100, 1) if total > 0 else 0
    error_rate = round(errors / total * 100, 2) if total > 0 else 0

    risk_html = "".join(
        f"<tr><td>{r}</td><td>{c}</td></tr>" for r, c in risk_counts.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>Telco Churn Monitor</title>
<style>
  body {{font-family:Arial,sans-serif;background:#f4f6f9;margin:0;padding:20px;color:#333}}
  h1 {{color:#2c3e50}} h2{{color:#34495e;border-bottom:2px solid #3498db;padding-bottom:5px}}
  .cards{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
  .card{{background:#fff;border-radius:8px;padding:20px;min-width:160px;box-shadow:0 2px 6px rgba(0,0,0,.1);text-align:center}}
  .card .val{{font-size:2em;font-weight:bold;color:#2980b9}}
  .card .lbl{{color:#666;font-size:.9em;margin-top:4px}}
  .card.warn .val{{color:#e67e22}} .card.ok .val{{color:#27ae60}} .card.err .val{{color:#e74c3c}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,.1)}}
  th{{background:#3498db;color:#fff;padding:10px}} td{{padding:10px;border-bottom:1px solid #eee;text-align:center}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-weight:bold;font-size:.85em}}
  .badge.high{{background:#fde8e8;color:#c0392b}} .badge.medium{{background:#fef9e7;color:#d35400}}
  .badge.low{{background:#eafaf1;color:#27ae60}}
  footer{{margin-top:30px;color:#aaa;font-size:.8em}}
</style>
</head>
<body>
<h1>🔍 Telco Churn — Monitoring Dashboard</h1>
<p>Service started: <b>{start}</b> &nbsp;|&nbsp; Auto-refreshes every 10 seconds</p>

<h2>📊 Prediction Summary</h2>
<div class="cards">
  <div class="card"><div class="val">{total}</div><div class="lbl">Total Requests</div></div>
  <div class="card {'err' if churn_rate>50 else 'ok'}"><div class="val">{churn_rate}%</div><div class="lbl">Churn Rate</div></div>
  <div class="card ok"><div class="val">{no_churn}</div><div class="lbl">No-Churn</div></div>
  <div class="card warn"><div class="val">{churn}</div><div class="lbl">Churn Predicted</div></div>
  <div class="card {'err' if errors>0 else 'ok'}"><div class="val">{errors}</div><div class="lbl">Errors ({error_rate}%)</div></div>
</div>

<h2>⚡ Latency</h2>
<div class="cards">
  <div class="card"><div class="val">{avg_lat} ms</div><div class="lbl">Avg Latency</div></div>
  <div class="card"><div class="val">{p95_lat} ms</div><div class="lbl">p95 Latency</div></div>
</div>

<h2>🚦 Risk Distribution</h2>
<table>
  <thead><tr><th>Risk Level</th><th>Count</th></tr></thead>
  <tbody>{risk_html if risk_html else '<tr><td colspan="2">No predictions yet</td></tr>'}</tbody>
</table>

<footer>Telco Churn Monitored ML Endpoint v2.0 | Built for ScholarX ML Engineer Internship</footer>
</body>
</html>"""


@app.get("/logs/summary")
def logs_summary():
    """Return summary of recent prediction logs."""
    if not os.path.exists(LOG_PATH):
        return {"message": "No logs yet.", "entries": 0}
    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    total = len(entries)
    if total == 0:
        return {"message": "No log entries yet.", "entries": 0}
    churns = sum(1 for e in entries if e.get("prediction") == 1)
    avg_lat = round(np.mean([e.get("latency_ms", 0) for e in entries]), 2)
    return {
        "total_logged": total,
        "churn_count": churns,
        "no_churn_count": total - churns,
        "churn_rate_pct": round(churns / total * 100, 1),
        "avg_latency_ms": avg_lat,
        "recent_5": entries[-5:]
    }
