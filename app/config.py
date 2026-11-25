# config.py - all settings and paths in one place

import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

SRC_DIR = os.path.join(BASE_DIR, "ml", "src")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "prediction_history.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "reports", "figures")

APP_TITLE = "UPI Fraud Detection"

WEAK_MODEL_WARNING = (
    "The model is weak: ROC-AUC is about 0.5 and PR-AUC is about 0.002, "
    "which is no better than random guessing. The data has almost no signal for fraud. "
    "Use this app to learn, not for real decisions."
)

# Choices for the drop-down lists
OPTIONS = {
    "transaction_type": ["Bill Payment", "P2M", "P2P", "Recharge"],
    "merchant_category": ["Education", "Entertainment", "Food", "Fuel", "Grocery",
                          "Healthcare", "Other", "Shopping", "Transport", "Utilities"],
    "transaction_status": ["SUCCESS", "FAILED"],
    "age_group": ["18-25", "26-35", "36-45", "46-55", "56+"],
    "sender_state": ["Andhra Pradesh", "Delhi", "Gujarat", "Karnataka", "Maharashtra",
                     "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"],
    "bank": ["Axis", "HDFC", "ICICI", "IndusInd", "Kotak", "PNB", "SBI", "Yes Bank"],
    "device_type": ["Android", "iOS", "Web"],
    "network_type": ["3G", "4G", "5G", "WiFi"],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                    "Saturday", "Sunday"],
}

# Columns the user can pick on the Fraud Analytics page
ANALYTICS_COLUMNS = ["transaction_type", "merchant_category", "sender_age_group",
                     "device_type", "network_type", "sender_bank", "sender_state",
                     "day_of_week", "hour_of_day"]