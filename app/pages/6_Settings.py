# Page 6 - Settings

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from utils import data_loader, helpers, model_loader
from components import sidebar

st.set_page_config(page_title="Settings", page_icon="⚙️")
sidebar.show_sidebar()
st.title("⚙️ Settings")

info = model_loader.get_model_info()

st.subheader("Fraud threshold")
st.write("A transaction is flagged as fraud when its probability is at least this value. "
         "A lower value flags more transactions.")
new_threshold = st.slider("Threshold", 0.1, 0.9, float(helpers.get_threshold()), 0.05)
if st.button("Save threshold"):
    helpers.set_threshold(new_threshold)
    st.success("Threshold saved: " + str(new_threshold) + " (for this session)")
if st.button("Reset to the model's value"):
    helpers.reset_threshold()
    st.success("Threshold reset to " + str(info["threshold"]))

st.subheader("Prediction history")
st.write("Saved predictions:", len(data_loader.load_history()))
if st.button("Clear history"):
    data_loader.clear_history()
    st.success("History cleared.")

st.subheader("Model information")
st.write("Model:", info["model_name"])
st.write("Best settings:", info["best_params"])
st.write("Saved threshold:", info["threshold"])
st.write("Input columns:", ", ".join(info["input_columns"]))