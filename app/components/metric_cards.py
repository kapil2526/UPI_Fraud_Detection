# metric_cards.py - rows of numbers (st.metric)

import streamlit as st
from utils import helpers


def dataset_cards(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud cases", f"{int(df['fraud_flag'].sum()):,}")
    c3.metric("Fraud rate", helpers.format_percent(df["fraud_flag"].mean()))
    c4.metric("Average amount", helpers.format_money(df["amount_inr"].mean()))


def history_cards(history):
    c1, c2 = st.columns(2)
    c1.metric("Predictions", len(history))
    c2.metric("Flagged as fraud", int(history["prediction"].sum()))


def model_cards(result):
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{result['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{result['pr_auc']:.4f}")
    c3.metric("Recall", f"{result['recall']:.2f}")
    c4, c5 = st.columns(2)
    c4.metric("Precision", f"{result['precision']:.4f}")
    c5.metric("F1-score", f"{result['f1']:.4f}")


def threshold_cards(m):
    a, b, c = st.columns(3)
    a.metric("Fraud caught", f"{m['true_positive']} of {m['true_positive'] + m['false_negative']}")
    b.metric("False alarms", f"{m['false_positive']:,}")
    c.metric("Precision", f"{m['precision']:.4f}")