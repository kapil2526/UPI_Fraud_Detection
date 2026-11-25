# Page 1 - Dashboard: a quick overview

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from utils import data_loader
from components import charts, metric_cards, sidebar

st.set_page_config(page_title="Dashboard", page_icon="🏠")
sidebar.show_sidebar()
st.title("🏠 Dashboard")

df = data_loader.need_data()

st.subheader("Dataset overview")
metric_cards.dataset_cards(df)

charts.monthly_charts(df)
charts.count_chart(df["transaction_type"], "Transactions by type")

st.subheader("Predictions made in this app")
history = data_loader.load_history()
if history.empty:
    st.info("No predictions yet. Go to the New Prediction page.")
else:
    metric_cards.history_cards(history)
    st.bar_chart(history["risk_level"].value_counts())