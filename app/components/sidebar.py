# sidebar.py - small info box shown in the sidebar of every page

import streamlit as st
from utils import data_loader, helpers, model_loader


def show_sidebar():
    info = model_loader.get_model_info()
    with st.sidebar:
        st.markdown("---")
        st.caption("Model: " + info["model_name"])
        st.caption("Threshold: " + str(round(helpers.get_threshold(), 2)))
        st.caption("Saved predictions: " + str(len(data_loader.load_history())))