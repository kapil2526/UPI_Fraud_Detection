# Page 2 - New Prediction: check one transaction, or a CSV file

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st
from config import OPTIONS
from utils import data_loader, helpers, model_loader, prediction
from components import prediction_result, sidebar

st.set_page_config(page_title="New Prediction", page_icon="🔍")
sidebar.show_sidebar()
st.title("🔍 New Prediction")
model_loader.need_model()

tab1, tab2 = st.tabs(["One transaction", "Upload a CSV file"])

# ---------------- Tab 1: one transaction ----------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (INR)", min_value=1.0, max_value=100000.0,
                                 value=500.0, step=50.0)
        hour = st.slider("Hour of day (0-23)", 0, 23, 12)
        day = st.selectbox("Day of week", OPTIONS["day_of_week"])
        ttype = st.selectbox("Transaction type", OPTIONS["transaction_type"])
        category = st.selectbox("Merchant category", OPTIONS["merchant_category"])
        status = st.selectbox("Transaction status", OPTIONS["transaction_status"])
    with col2:
        sender_age = st.selectbox("Sender age group", OPTIONS["age_group"])
        receiver_age = st.selectbox("Receiver age group", OPTIONS["age_group"])
        state = st.selectbox("Sender state", OPTIONS["sender_state"])
        sender_bank = st.selectbox("Sender bank", OPTIONS["bank"])
        receiver_bank = st.selectbox("Receiver bank", OPTIONS["bank"])
        device = st.selectbox("Device type", OPTIONS["device_type"])
        network = st.selectbox("Network type", OPTIONS["network_type"])

    if st.button("Check for fraud", type="primary"):
        threshold = helpers.get_threshold()
        transaction = prediction.make_transaction(
            amount, hour, day, ttype, category, status, sender_age, receiver_age,
            state, sender_bank, receiver_bank, device, network)
        result = prediction.check_one(transaction, threshold)
        data_loader.save_to_history(transaction, result)
        prediction_result.show_result(result, threshold)

# ---------------- Tab 2: CSV file ----------------
with tab2:
    st.write("Your file needs these columns:")
    st.code(", ".join(model_loader.get_model_info()["input_columns"]))
    st.caption("If is_weekend is missing, it is made from day_of_week.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        try:
            result = prediction.check_many(data, helpers.get_threshold())
            c1, c2 = st.columns(2)
            c1.metric("Rows checked", len(result))
            c2.metric("Flagged as fraud", int(result["prediction"].sum()))
            st.bar_chart(result["risk_level"].value_counts())
            st.dataframe(result.head(100))
        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")