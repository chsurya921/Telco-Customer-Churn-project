"""
Task 3: Feature Validation Pipeline
Validates incoming data before it reaches the model.
Catches: missing fields, wrong types, out-of-range values, distribution drift.
"""

import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Schema definition ────────────────────────────────────────────────────────
FEATURE_SCHEMA = {
    "gender":           {"type": str,   "allowed": ["Male", "Female"]},
    "SeniorCitizen":    {"type": int,   "min": 0, "max": 1},
    "Partner":          {"type": str,   "allowed": ["Yes", "No"]},
    "Dependents":       {"type": str,   "allowed": ["Yes", "No"]},
    "tenure":           {"type": int,   "min": 0, "max": 100},
    "PhoneService":     {"type": str,   "allowed": ["Yes", "No"]},
    "MultipleLines":    {"type": str,   "allowed": ["Yes", "No", "No phone service"]},
    "InternetService":  {"type": str,   "allowed": ["DSL", "Fiber optic", "No"]},
    "OnlineSecurity":   {"type": str,   "allowed": ["Yes", "No", "No internet service"]},
    "OnlineBackup":     {"type": str,   "allowed": ["Yes", "No", "No internet service"]},
    "DeviceProtection": {"type": str,   "allowed": ["Yes", "No", "No internet service"]},
    "TechSupport":      {"type": str,   "allowed": ["Yes", "No", "No internet service"]},
    "StreamingTV":      {"type": str,   "allowed": ["Yes", "No", "No internet service"]},
    "StreamingMovies":  {"type": str,   "allowed": ["Yes", "No", "No internet service"]},
    "Contract":         {"type": str,   "allowed": ["Month-to-month", "One year", "Two year"]},
    "PaperlessBilling": {"type": str,   "allowed": ["Yes", "No"]},
    "PaymentMethod":    {"type": str,   "allowed": [
                            "Bank transfer (automatic)", "Credit card (automatic)",
                            "Electronic check", "Mailed check"
                        ]},
    "MonthlyCharges":   {"type": float, "min": 0.0,  "max": 200.0},
    "TotalCharges":     {"type": float, "min": 0.0,  "max": 10000.0},
}

# Reference statistics from training data (Telco dataset baseline)
REFERENCE_STATS = {
    "tenure":          {"mean": 32.37, "std": 24.56},
    "MonthlyCharges":  {"mean": 64.76, "std": 30.09},
    "TotalCharges":    {"mean": 2283.3, "std": 2266.77},
    "SeniorCitizen":   {"mean": 0.162, "std": 0.369},
}

DRIFT_Z_THRESHOLD = 3.0   # flag if value is >3 std from training mean


# ── Validation result containers ─────────────────────────────────────────────
@dataclass
class FieldError:
    field: str
    error_type: str   # missing | type_error | out_of_range | invalid_category | drift
    message: str
    value: Any = None


@dataclass
class ValidationReport:
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_valid: bool = True
    errors: List[FieldError] = field(default_factory=list)
    warnings: List[FieldError] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)

    def add_error(self, field: str, error_type: str, message: str, value=None):
        self.errors.append(FieldError(field, error_type, message, value))
        self.is_valid = False

    def add_warning(self, field: str, error_type: str, message: str, value=None):
        self.warnings.append(FieldError(field, error_type, message, value))

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "is_valid": self.is_valid,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "errors": [
                {"field": e.field, "type": e.error_type,
                 "message": e.message, "value": e.value}
                for e in self.errors
            ],
            "warnings": [
                {"field": e.field, "type": e.error_type,
                 "message": e.message, "value": e.value}
                for e in self.warnings
            ],
            "passed_checks": self.passed_checks,
        }


# ── Core validator ────────────────────────────────────────────────────────────
class FeatureValidator:
    """
    Validates a single customer record or a batch DataFrame.
    Performs: presence check, type check, range/category check, drift check.
    """

    def __init__(self, schema=None, reference_stats=None, drift_threshold=DRIFT_Z_THRESHOLD):
        self.schema = schema or FEATURE_SCHEMA
        self.reference_stats = reference_stats or REFERENCE_STATS
        self.drift_threshold = drift_threshold

    # ── Single record ────────────────────────────────────────────────────────
    def validate_record(self, record: Dict[str, Any]) -> ValidationReport:
        report = ValidationReport()

        for feat, rules in self.schema.items():
            # 1. Presence check
            if feat not in record or record[feat] is None:
                report.add_error(feat, "missing", f"Required field '{feat}' is missing or null.")
                continue

            val = record[feat]

            # 2. Type check (allow int when float expected)
            expected_type = rules["type"]
            if expected_type == float and isinstance(val, int):
                val = float(val)
                record[feat] = val
            if not isinstance(val, expected_type):
                report.add_error(feat, "type_error",
                    f"Expected {expected_type.__name__}, got {type(val).__name__}.", val)
                continue

            # 3. Category check
            if "allowed" in rules:
                if val not in rules["allowed"]:
                    report.add_error(feat, "invalid_category",
                        f"'{val}' not in allowed values: {rules['allowed']}.", val)
                continue

            # 4. Range check
            if "min" in rules and val < rules["min"]:
                report.add_error(feat, "out_of_range",
                    f"Value {val} < minimum {rules['min']}.", val)
                continue
            if "max" in rules and val > rules["max"]:
                report.add_error(feat, "out_of_range",
                    f"Value {val} > maximum {rules['max']}.", val)
                continue

            report.passed_checks.append(feat)

            # 5. Drift check (warning only)
            if feat in self.reference_stats:
                ref = self.reference_stats[feat]
                z = abs(val - ref["mean"]) / (ref["std"] + 1e-9)
                if z > self.drift_threshold:
                    report.add_warning(feat, "drift",
                        f"Value {val} is {z:.1f} std deviations from training mean "
                        f"({ref['mean']:.2f}). Possible distribution drift.", val)

        return report

    # ── Batch validation ─────────────────────────────────────────────────────
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Returns:
            valid_df    - rows that passed all checks
            invalid_df  - rows that failed one or more checks
            reports     - list of ValidationReport dicts (one per row)
        """
        valid_mask = []
        reports = []

        for idx, row in df.iterrows():
            record = row.to_dict()
            report = self.validate_record(record)
            reports.append({"row_index": idx, **report.to_dict()})
            valid_mask.append(report.is_valid)

        valid_mask_series = pd.Series(valid_mask, index=df.index)
        valid_df = df[valid_mask_series].copy()
        invalid_df = df[~valid_mask_series].copy()

        logger.info(
            f"Validation complete: {len(valid_df)} valid, "
            f"{len(invalid_df)} invalid out of {len(df)} rows."
        )
        return valid_df, invalid_df, reports

    # ── Dataset-level drift detection ────────────────────────────────────────
    def detect_drift(self, df: pd.DataFrame) -> Dict:
        """Compare numeric columns of incoming batch vs training reference stats."""
        drift_report = {"drifted_features": [], "stable_features": []}

        for feat, ref in self.reference_stats.items():
            if feat not in df.columns:
                continue
            col = pd.to_numeric(df[feat], errors="coerce").dropna()
            if len(col) == 0:
                continue
            batch_mean = col.mean()
            batch_std = col.std()
            z = abs(batch_mean - ref["mean"]) / (ref["std"] / np.sqrt(len(col)) + 1e-9)

            entry = {
                "feature": feat,
                "training_mean": ref["mean"],
                "batch_mean": round(batch_mean, 4),
                "training_std": ref["std"],
                "batch_std": round(batch_std, 4),
                "z_score": round(z, 2),
            }

            if z > self.drift_threshold:
                entry["status"] = "DRIFT_DETECTED"
                drift_report["drifted_features"].append(entry)
                logger.warning(f"Drift detected in '{feat}': z={z:.2f}")
            else:
                entry["status"] = "stable"
                drift_report["stable_features"].append(entry)

        return drift_report


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    validator = FeatureValidator()

    # ── Example 1: valid record ──────────────────────────────────────────────
    good_record = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 24, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.9, "TotalCharges": 1340.0
    }

    report = validator.validate_record(good_record)
    print("=== Valid Record ===")
    print(json.dumps(report.to_dict(), indent=2))

    # ── Example 2: bad record ────────────────────────────────────────────────
    bad_record = {
        "gender": "Unknown",           # invalid category
        "SeniorCitizen": 5,            # out of range
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -3,                  # negative
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
        "MonthlyCharges": 55.9,
        # TotalCharges missing
    }

    report2 = validator.validate_record(bad_record)
    print("\n=== Invalid Record ===")
    print(json.dumps(report2.to_dict(), indent=2))
