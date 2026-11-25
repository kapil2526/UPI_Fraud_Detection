# predict.py - loads the saved model and predicts fraud for new transactions

import os
import sys
import json
import joblib
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SRC_DIR))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "best_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "model_info.json")

sys.path.append(SRC_DIR)   # the saved model needs feature_engineering.py

_model = None   # the model is loaded only once


def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_risk_level(probability, threshold):
    """Low / Medium / High from the fraud probability."""
    if probability >= threshold:
        return "High"
    if probability >= threshold * 0.8:
        return "Medium"
    return "Low"


def predict_many(df, threshold=None):
    """df = table of transactions. Returns the table with 3 new columns.
    threshold = custom cut-off (if None, the saved one is used)."""
    info = load_info()
    columns = info["input_columns"]
    if threshold is None:
        threshold = info["threshold"]

    df = df.copy()
    # is_weekend can be made from day_of_week if it is missing
    if "is_weekend" not in df.columns and "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    proba = load_model().predict_proba(df[columns])[:, 1]
    df["fraud_probability"] = proba
    df["prediction"] = (proba >= threshold).astype(int)
    df["risk_level"] = [get_risk_level(p, threshold) for p in proba]
    return df


def predict_one(transaction, threshold=None):
    """transaction = dictionary with one value per input column."""
    result = predict_many(pd.DataFrame([transaction]), threshold).iloc[0]
    return {
        "fraud_probability": float(result["fraud_probability"]),
        "prediction": int(result["prediction"]),
        "risk_level": result["risk_level"],
    }

if __name__ == "__main__":
    example = {
        "amount_inr": 25000, "hour_of_day": 2, "transaction_type": "P2P",
        "merchant_category": "Other", "transaction_status": "SUCCESS",
        "sender_age_group": "18-25", "receiver_age_group": "56+",
        "sender_state": "Delhi", "sender_bank": "SBI", "receiver_bank": "HDFC",
        "device_type": "Android", "network_type": "4G", "day_of_week": "Sunday",
    }
    print(predict_one(example))