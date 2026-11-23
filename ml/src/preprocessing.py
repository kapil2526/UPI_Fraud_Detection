"""
preprocessing.py
----------------
Turns the cleaned data into train/test files that a model can use.

Steps:
1. Split into X (inputs) and y (target = fraud_flag)
2. Stratified train/test split (keeps the same fraud % in both)
3. One-hot encode text columns, scale number columns
   (the pipeline is FIT on training data ONLY -> no data leakage)
4. Save X_train, X_test, y_train, y_test and the fitted pipeline

NOTE: No SMOTE here. Person 2 does class imbalance on the TRAINING data only.
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- File paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PIPELINE_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "preprocessing_pipeline.pkl")

# ---------- Column settings ----------
TARGET = "fraud_flag"
# Columns that do not help the model
DROP_COLUMNS = ["transaction_id", "timestamp"]
# Number columns -> scaled
NUMERIC_COLUMNS = ["amount_inr", "hour_of_day"]
# Text columns -> one-hot encoded
CATEGORICAL_COLUMNS = [
    "transaction_type", "merchant_category", "transaction_status",
    "sender_age_group", "receiver_age_group", "sender_state",
    "sender_bank", "receiver_bank", "device_type", "network_type",
    "day_of_week",
]
# Already 0/1 columns -> left as they are
BINARY_COLUMNS = ["is_weekend"]


def load_cleaned_data(path=CLEAN_PATH):
    """Read the cleaned csv."""
    return pd.read_csv(path)


def split_x_y(df):
    """Separate inputs (X) and target (y)."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    return X, y


def build_preprocessor():
    """Create the ColumnTransformer (not fitted yet)."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),
            ("bin", "passthrough", BINARY_COLUMNS),
        ]
    )
    # Put it inside a Pipeline so it is easy to save and reuse
    pipeline = Pipeline(steps=[("preprocessor", column_transformer)])
    return pipeline


def split_train_test(X, y, test_size=0.2, random_state=42):
    """Stratified split so train and test both keep the same fraud %."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def transform_to_dataframe(pipeline, X):
    """Apply the fitted pipeline and return a DataFrame with column names."""
    array = pipeline.transform(X)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    # Remove the "num__" / "cat__" / "bin__" prefix to keep names short
    names = [n.split("__", 1)[1] for n in names]
    return pd.DataFrame(array, columns=names, index=X.index)


def save_pipeline(pipeline, path=PIPELINE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path=PIPELINE_PATH):
    return joblib.load(path)


def preprocess_new_data(new_df, path=PIPELINE_PATH):
    """Use the saved pipeline on NEW data (for future predictions).
    new_df must have the same input columns as X (without fraud_flag)."""
    pipeline = load_pipeline(path)
    new_df = new_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS]
    return transform_to_dataframe(pipeline, new_df)


def main():
    df = load_cleaned_data()
    X, y = split_x_y(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_preprocessor()
    pipeline.fit(X_train)  # learn ONLY from training data

    X_train_final = transform_to_dataframe(pipeline, X_train)
    X_test_final = transform_to_dataframe(pipeline, X_test)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train_final.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test_final.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
    save_pipeline(pipeline)

    print("X_train:", X_train_final.shape, " X_test:", X_test_final.shape)
    print("Fraud in train:", y_train.sum(), f"({y_train.mean():.4%})")
    print("Fraud in test :", y_test.sum(), f"({y_test.mean():.4%})")
    print("Saved files in", PROCESSED_DIR)


if __name__ == "__main__":
    main()