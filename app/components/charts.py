# charts.py - charts and tables used by the pages

import pandas as pd
import streamlit as st


def monthly_charts(df):
    st.subheader("Transactions per month")
    st.bar_chart(df.groupby(df["timestamp"].dt.month).size())
    st.subheader("Fraud cases per month")
    fraud = df[df["fraud_flag"] == 1]
    st.bar_chart(fraud.groupby(fraud["timestamp"].dt.month).size())


def count_chart(series, title):
    st.subheader(title)
    st.bar_chart(series.value_counts())


def fraud_rate_by(df, column):
    """Chart + table of the fraud rate for each value of one column."""
    table = df.groupby(column)["fraud_flag"].agg(["sum", "count", "mean"])
    table["fraud_rate_%"] = (table["mean"] * 100).round(3)
    table = table.rename(columns={"sum": "fraud_cases", "count": "transactions"})
    st.bar_chart(table["fraud_rate_%"])
    st.dataframe(table[["transactions", "fraud_cases", "fraud_rate_%"]])


def fraud_rate_by_amount(df):
    """Chart + table of the fraud rate for amount ranges."""
    bins = [0, 100, 500, 1000, 2000, 5000, 10000, 100000]
    ranges = pd.cut(df["amount_inr"], bins=bins)
    table = df.groupby(ranges, observed=True)["fraud_flag"].agg(["sum", "count", "mean"])
    table.index = table.index.astype(str)
    table["fraud_rate_%"] = (table["mean"] * 100).round(3)
    table = table.rename(columns={"sum": "fraud_cases", "count": "transactions"})
    st.bar_chart(table["fraud_rate_%"])
    st.dataframe(table[["transactions", "fraud_cases", "fraud_rate_%"]])