# helpers.py - small helper functions

import streamlit as st
from utils import model_loader


def get_threshold():
    """The threshold chosen on the Settings page (or the saved one)."""
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = model_loader.get_model_info()["threshold"]
    return st.session_state["threshold"]


def set_threshold(value):
    st.session_state["threshold"] = value


def reset_threshold():
    st.session_state["threshold"] = model_loader.get_model_info()["threshold"]


def format_money(value):
    return f"Rs {value:,.0f}"


def format_percent(value, digits=2):
    return f"{value:.{digits}%}"