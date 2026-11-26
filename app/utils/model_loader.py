# model_loader.py - gives the app access to the saved model
# model_loader.py - gives the app access to the saved model

import os
import sys
import streamlit as st
from config import SRC_DIR, MODEL_PATH

sys.path.append(SRC_DIR)   # so we can import ml/src/predict.py
import predict


def need_model():
    """Stop the page with a clear message if the model file is missing."""
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found: ml/artifacts/best_model.pkl")
        st.info("The model file must be committed to GitHub. "
                "Locally you can create it with: python ml/src/train_model.py")
        st.stop()


def get_model():
    """The saved pipeline (it is loaded only once)."""
    need_model()
    return predict.load_model()


def get_model_info():
    """The content of ml/artifacts/model_info.json."""
    return predict.load_info()