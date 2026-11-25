# streamlit_app.py - main page. Run:  python -m streamlit run app/streamlit_app.py

import streamlit as st
from config import APP_TITLE, WEAK_MODEL_WARNING
from components import sidebar

st.set_page_config(page_title=APP_TITLE, page_icon="🛡️")
sidebar.show_sidebar()

st.title("🛡️ " + APP_TITLE)
st.write("This app uses a machine-learning model to check UPI transactions for fraud.")

st.subheader("Pages")
st.page_link("pages/1_Dashboard.py", label="Dashboard", icon="🏠")
st.page_link("pages/2_New_Prediction.py", label="New Prediction", icon="🔍")
st.page_link("pages/3_Transaction_History.py", label="Transaction History", icon="🕒")
st.page_link("pages/4_Fraud_Analytics.py", label="Fraud Analytics", icon="📈")
st.page_link("pages/5_Model_Performance.py", label="Model Performance", icon="📊")
st.page_link("pages/6_Settings.py", label="Settings", icon="⚙️")

st.warning(WEAK_MODEL_WARNING)