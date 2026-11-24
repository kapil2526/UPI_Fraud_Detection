# feature_engineering.py - makes new columns and the encoder

import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import preprocessing as prep

# Columns the model needs as input (14 columns)
INPUT_COLUMNS = prep.NUMERIC_COLUMNS + prep.CATEGORICAL_COLUMNS + prep.BINARY_COLUMNS

# Columns after we add new features
NUMBER_COLUMNS = ["amount_inr", "hour_of_day", "log_amount"]
TEXT_COLUMNS = prep.CATEGORICAL_COLUMNS
ZERO_ONE_COLUMNS = ["is_weekend", "is_night", "is_high_amount", "same_bank", "same_age_group"]


def add_features(df):
    """Add 5 new columns (uses only the same row, so no data leakage)."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount_inr"])                       # smoother amount
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)        # night payment
    df["is_high_amount"] = (df["amount_inr"] > 3500).astype(int)        # big payment
    df["same_bank"] = (df["sender_bank"] == df["receiver_bank"]).astype(int)
    df["same_age_group"] = (df["sender_age_group"] == df["receiver_age_group"]).astype(int)
    return df


def load_train_test():
    """Same train/test split as Person 1."""
    df = prep.load_cleaned_data()
    X, y = prep.split_x_y(df)
    X = X[INPUT_COLUMNS]
    return prep.split_train_test(X, y)


def build_encoder():
    """Scale numbers and one-hot encode text."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMBER_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), TEXT_COLUMNS),
        ("bin", "passthrough", ZERO_ONE_COLUMNS),
    ])