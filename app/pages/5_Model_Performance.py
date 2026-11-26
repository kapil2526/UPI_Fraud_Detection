# Page 5 - Model Performance

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import streamlit as st
from config import FIGURE_DIR, WEAK_MODEL_WARNING
from utils import helpers, model_loader, prediction
from components import metric_cards, sidebar

st.set_page_config(page_title="Model Performance", page_icon="📊")
sidebar.show_sidebar()
st.title("📊 Model Performance")
model_loader.need_model()

info = model_loader.get_model_info()
st.write("Model:", info["model_name"])
metric_cards.model_cards(info["test_result"])
st.warning(WEAK_MODEL_WARNING + " Accuracy is not shown because it would be misleading: "
           "99.8% of the transactions are legit.")

# ----- Try different thresholds on the TEST data -----
st.subheader("Try a different threshold")
t = st.slider("Threshold", 0.1, 0.9, float(helpers.get_threshold()), 0.05)
m = prediction.get_metrics_at(t)
metric_cards.threshold_cards(m)
st.table(pd.DataFrame([[m["true_negative"], m["false_positive"]],
                       [m["false_negative"], m["true_positive"]]],
                      index=["Actual legit", "Actual fraud"],
                      columns=["Predicted legit", "Predicted fraud"]))

# ----- Saved plots -----
st.subheader("Curves")
path = os.path.join(FIGURE_DIR, "20_roc_pr_curves.png")
if os.path.exists(path):
    st.image(path)
else:
    st.info("Image not found. Run: python ml/src/evaluate_model.py")