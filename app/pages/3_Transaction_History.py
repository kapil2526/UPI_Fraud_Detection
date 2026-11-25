# Page 3 - Transaction History: all predictions made in this app

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from utils import data_loader
from components import sidebar

st.set_page_config(page_title="Transaction History", page_icon="🕒")
sidebar.show_sidebar()
st.title("🕒 Transaction History")

history = data_loader.load_history()
if history.empty:
    st.info("No predictions yet. Go to the New Prediction page and check a transaction.")
    st.stop()

levels = st.multiselect("Risk level", ["Low", "Medium", "High"],
                        default=["Low", "Medium", "High"])
shown = history[history["risk_level"].isin(levels)]
shown = shown.sort_values("time", ascending=False)   # newest first

c1, c2 = st.columns(2)
c1.metric("Predictions shown", len(shown))
c2.metric("Flagged as fraud", int(shown["prediction"].sum()))

st.dataframe(shown)
st.download_button("Download history", shown.to_csv(index=False),
                   file_name="prediction_history.csv", mime="text/csv")