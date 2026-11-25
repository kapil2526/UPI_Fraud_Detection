# prediction_result.py - shows the result of one prediction

import streamlit as st


def show_result(result, threshold):
    probability = result["fraud_probability"]
    st.metric("Fraud probability", f"{probability:.1%}")
    st.progress(min(probability, 1.0))

    if result["risk_level"] == "High":
        st.error("🚨 High risk: this transaction is flagged as possible fraud.")
    elif result["risk_level"] == "Medium":
        st.warning("⚠️ Medium risk: please review this transaction.")
    else:
        st.success("✅ Low risk: this transaction looks normal.")
    st.caption("Threshold used: " + str(round(threshold, 2)) +
               ". The model is weak, do not use it for real decisions.")