# test_data.py - checks the cleaned data and the train/test files
# Run:  python -m pytest tests

import os
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")
PROCESSED = os.path.join(ROOT, "data", "processed")

pytestmark = pytest.mark.skipif(not os.path.exists(CLEAN),
                                reason="run python ml/src/data_cleaning.py first")


@pytest.fixture(scope="module")
def clean():
    return pd.read_csv(CLEAN)


def test_no_missing_values(clean):
    assert clean.isnull().sum().sum() == 0


def test_no_duplicate_ids(clean):
    assert clean["transaction_id"].duplicated().sum() == 0


def test_target_is_zero_or_one(clean):
    assert set(clean["fraud_flag"].unique()) <= {0, 1}


def test_fraud_is_rare(clean):
    assert 0 < clean["fraud_flag"].mean() < 0.01


def test_amount_is_positive(clean):
    assert (clean["amount_inr"] > 0).all()


def test_hour_range(clean):
    assert clean["hour_of_day"].between(0, 23).all()


def test_column_names_are_clean(clean):
    for name in ["transaction_id", "transaction_type", "amount_inr", "fraud_flag"]:
        assert name in clean.columns


@pytest.mark.skipif(not os.path.exists(os.path.join(PROCESSED, "X_train.csv")),
                    reason="run python ml/src/preprocessing.py first")
def test_train_test_files():
    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv")).iloc[:, 0]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == list(X_test.columns)
    assert "fraud_flag" not in X_train.columns              # no leakage
    assert abs(y_train.mean() - y_test.mean()) < 0.0005     # stratified split