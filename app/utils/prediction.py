# prediction.py - functions the pages use to make predictions

import streamlit as st
import pandas as pd
from utils import model_loader
import predict
import evaluate_model


def make_transaction(amount, hour, day, ttype, category, status, sender_age,
                     receiver_age, state, sender_bank, receiver_bank, device, network):
    """Put the form values in a dictionary (the format the model needs)."""
    return {
        "amount_inr": amount, "hour_of_day": hour,
        "transaction_type": ttype, "merchant_category": category,
        "transaction_status": status, "sender_age_group": sender_age,
        "receiver_age_group": receiver_age, "sender_state": state,
        "sender_bank": sender_bank, "receiver_bank": receiver_bank,
        "device_type": device, "network_type": network,
        "day_of_week": day,
        "is_weekend": 1 if day in ["Saturday", "Sunday"] else 0,
    }


def check_one(transaction, threshold):
    """Predict one transaction. Returns probability, prediction and risk level."""
    return predict.predict_one(transaction, threshold)


def check_many(df, threshold):
    """Predict many transactions. Raises ValueError if columns are missing."""
    return predict.predict_many(df, threshold)


@st.cache_data
def get_test_scores():
    """True labels and fraud probabilities of the test data (calculated once)."""
    return evaluate_model.get_test_scores()


def get_metrics_at(threshold):
    """Precision, recall, confusion matrix numbers ... at one threshold."""
    y_true, proba = get_test_scores()
    return evaluate_model.get_metrics(y_true, proba, threshold)