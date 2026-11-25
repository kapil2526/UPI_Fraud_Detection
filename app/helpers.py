# helpers.py - shared code for all pages

import os
import sys
import datetime
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
sys.path.append(os.path.join(BASE_DIR, "ml", "src"))   # so we can import predict.py

import predict

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "prediction_history.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Choices for the drop-down lists
OPTIONS = {
    "transaction_type": ["Bill Payment", "P2M", "P2P", "Recharge"],
    "merchant_category": ["Education", "Entertainment", "Food", "Fuel", "Grocery",
                          "Healthcare", "Other", "Shopping", "Transport", "Utilities"],
    "transaction_status": ["SUCCESS", "FAILED"],
    "age_group": ["18-25", "26-35", "36-45", "46-55", "56+"],
    "sender_state": ["Andhra Pradesh", "Delhi", "Gujarat", "Karnataka", "Maharashtra",
                     "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"],
    "bank": ["Axis", "HDFC", "ICICI", "IndusInd", "Kotak", "PNB", "SBI", "Yes Bank"],
    "device_type": ["Android", "iOS", "Web"],
    "network_type": ["3G", "4G", "5G", "WiFi"],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                    "Saturday", "Sunday"],
}


def get_threshold():
    """The threshold chosen on the Settings page (or the saved one)."""
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = predict.load_info()["threshold"]
    return st.session_state["threshold"]


@st.cache_data
def load_data():
    """Read the cleaned data. Returns None if the file is missing."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def need_data():
    """Return the data, or show a message and stop the page."""
    df = load_data()
    if df is None:
        st.error("File data/processed/cleaned_data.csv not found.")
        st.info("Run this command first: python ml/src/data_cleaning.py")
        st.stop()
    return df


def save_to_history(transaction, result):
    """Add one prediction to the history file."""
    row = dict(transaction)
    row.update(result)
    row["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([row])
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if os.path.exists(HISTORY_PATH):
        new_row = pd.concat([pd.read_csv(HISTORY_PATH), new_row], ignore_index=True)
    new_row.to_csv(HISTORY_PATH, index=False)


def load_history():
    """Read the history file (empty table if there is none)."""
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)