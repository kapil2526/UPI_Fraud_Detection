# test_prediction.py - checks the prediction functions

import os
import sys
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "ml", "src"))
MODEL = os.path.join(ROOT, "ml", "artifacts", "best_model.pkl")

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="best_model.pkl not found")

EXAMPLE = {
    "amount_inr": 250, "hour_of_day": 14, "transaction_type": "P2M",
    "merchant_category": "Food", "transaction_status": "SUCCESS",
    "sender_age_group": "26-35", "receiver_age_group": "26-35",
    "sender_state": "Maharashtra", "sender_bank": "HDFC", "receiver_bank": "HDFC",
    "device_type": "Android", "network_type": "5G", "day_of_week": "Tuesday",
}


def test_predict_one_returns_three_values():
    import predict
    result = predict.predict_one(EXAMPLE)
    assert set(result) == {"fraud_probability", "prediction", "risk_level"}
    assert 0 <= result["fraud_probability"] <= 1
    assert result["prediction"] in (0, 1)
    assert result["risk_level"] in ("Low", "Medium", "High")


def test_predict_many_adds_columns():
    import predict
    df = pd.DataFrame([EXAMPLE, EXAMPLE])
    out = predict.predict_many(df)
    assert len(out) == 2
    for col in ["fraud_probability", "prediction", "risk_level"]:
        assert col in out.columns


def test_is_weekend_is_made_from_day():
    import predict
    out = predict.predict_many(pd.DataFrame([dict(EXAMPLE, day_of_week="Sunday")]))
    assert out["prediction"].iloc[0] in (0, 1)


def test_missing_columns_give_error():
    import predict
    with pytest.raises(ValueError):
        predict.predict_one({"amount_inr": 100})


def test_unknown_category_does_not_crash():
    import predict
    result = predict.predict_one(dict(EXAMPLE, merchant_category="NewCategory"))
    assert 0 <= result["fraud_probability"] <= 1


def test_threshold_changes_the_prediction():
    import predict
    assert predict.predict_one(EXAMPLE, threshold=0.0)["prediction"] == 1
    assert predict.predict_one(EXAMPLE, threshold=1.1)["prediction"] == 0


def test_app_make_transaction():
    sys.path.append(os.path.join(ROOT, "app"))
    from utils import prediction
    t = prediction.make_transaction(100, 3, "Sunday", "P2P", "Other", "SUCCESS", "18-25",
                                    "26-35", "Delhi", "SBI", "HDFC", "Android", "4G")
    assert t["is_weekend"] == 1
    assert t["amount_inr"] == 100