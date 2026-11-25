# test_model.py - checks the saved model and its info file

import os
import sys
import json
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "ml", "src"))
MODEL = os.path.join(ROOT, "ml", "artifacts", "best_model.pkl")
INFO = os.path.join(ROOT, "ml", "artifacts", "model_info.json")


def test_info_file_has_needed_keys():
    assert os.path.exists(INFO)
    info = json.load(open(INFO))
    for key in ["model_name", "threshold", "input_columns", "test_result"]:
        assert key in info
    assert len(info["input_columns"]) == 14


@pytest.mark.skipif(not os.path.exists(MODEL), reason="best_model.pkl not found")
def test_model_loads_and_predicts():
    import predict
    model = predict.load_model()
    info = predict.load_info()
    clean = pd.read_csv(os.path.join(ROOT, "data", "processed", "cleaned_data.csv")).head(10)
    proba = model.predict_proba(clean[info["input_columns"]])
    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_test_metrics_exist():
    result = json.load(open(INFO))["test_result"]
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert key in result