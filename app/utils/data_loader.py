# data_loader.py - reads the data and keeps the prediction history

import os
import datetime
import pandas as pd
import streamlit as st
from config import DATA_PATH, HISTORY_PATH


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
    """Read the history file (an empty table if there is none)."""
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)