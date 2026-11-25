# Page 4 - Fraud Analytics: where does fraud happen in the data?

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from config import ANALYTICS_COLUMNS
from utils import data_loader, helpers
from components import charts, sidebar

st.set_page_config(page_title="Fraud Analytics", page_icon="📈")
sidebar.show_sidebar()
st.title("📈 Fraud Analytics")

df = data_loader.need_data()

column = st.selectbox("Show fraud rate by", ANALYTICS_COLUMNS)
charts.fraud_rate_by(df, column)
st.caption("Overall fraud rate: " + helpers.format_percent(df["fraud_flag"].mean(), 3) +
           ". Groups with few fraud cases change a lot by chance.")

st.subheader("Fraud rate by amount")
charts.fraud_rate_by_amount(df)

st.info("Almost all groups are close to the overall rate (about 0.19%). "
        "The features have very little link to fraud.")