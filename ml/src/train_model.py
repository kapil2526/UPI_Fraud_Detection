# train_model.py - compare models, tune the best one, save it
# Run:  python ml/src/train_model.py

import os
import sys
import json
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import feature_engineering as fe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "best_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "model_info.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


def prepare_data():
    """Add features, encode, scale and pick the best 30 features.
    Everything is learned from TRAIN data only."""
    X_train, X_test, y_train, y_test = fe.load_train_test()

    add_step = FunctionTransformer(fe.add_features).fit(X_train)
    encoder = fe.build_encoder()
    train_array = encoder.fit_transform(add_step.transform(X_train))
    test_array = encoder.transform(add_step.transform(X_test))

    selector = SelectKBest(f_classif, k=30).fit(train_array, y_train)

    return {
        "add_step": add_step, "encoder": encoder, "selector": selector,
        "X_train": selector.transform(train_array), "X_test": selector.transform(test_array),
        "X_test_raw": X_test, "y_train": y_train.values, "y_test": y_test.values,
    }


def get_model(name, use_weight):
    """use_weight=True gives fraud rows more importance."""
    w = "balanced" if use_weight else None
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight=w)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, class_weight=w)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=50,
                                      n_jobs=-1, class_weight=w, random_state=42)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1, random_state=42,
                             scale_pos_weight=500 if use_weight else 1)


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def compare_models(data):
    """Try 4 models x 3 ways of handling imbalance on a validation split."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        data["X_train"], data["y_train"], test_size=0.2, stratify=data["y_train"], random_state=42)

    rows = []
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]:
        for method in ["nothing", "class_weight", "smote"]:
            X_fit, y_fit = X_tr, y_tr
            if method == "smote":   # only on training part, never on validation/test
                X_fit, y_fit = SMOTE(sampling_strategy=0.1, random_state=42).fit_resample(X_tr, y_tr)
            model = get_model(name, use_weight=(method == "class_weight"))
            model.fit(X_fit, y_fit)
            row = {"model": name, "method": method}
            row.update(evaluate(y_val, model.predict_proba(X_val)[:, 1]))
            rows.append(row)
            print("done:", name, method)
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


SMALL_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"max_depth": [3, 5, 8], "min_samples_leaf": [50, 100, 300]},
    "Random Forest": {"max_depth": [4, 6, 8], "min_samples_leaf": [20, 50]},
    "XGBoost": {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]},
}


def tune_best_model(data, comparison):
    """Tune the top model with GridSearchCV (3-fold cross-validation on train data)."""
    name = comparison.loc[0, "model"]
    print("tuning:", name)
    search = GridSearchCV(get_model(name, use_weight=True), SMALL_GRIDS[name],
                          scoring="average_precision", cv=3)
    search.fit(data["X_train"], data["y_train"])
    return name, search.best_estimator_, search.best_params_, search.best_score_


def save_model(data, name, model, params, cv_score, comparison):
    """Save ONE pipeline: new features -> encode -> select -> model."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    pipeline = Pipeline([("add", data["add_step"]), ("encode", data["encoder"]),
                         ("select", data["selector"]), ("model", model)])
    proba = pipeline.predict_proba(data["X_test_raw"])[:, 1]
    test_result = evaluate(data["y_test"], proba)   # test data used only here, once

    joblib.dump(pipeline, MODEL_PATH)
    comparison.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

    info = {"model_name": name, "best_params": params, "cv_pr_auc": cv_score,
            "threshold": 0.5, "input_columns": fe.INPUT_COLUMNS, "test_result": test_result}
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2, default=float)
    return test_result


def predict_fraud(df):
    """For Person 3: df must have the columns in fe.INPUT_COLUMNS."""
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(df[fe.INPUT_COLUMNS])[:, 1]
    return pd.DataFrame({"fraud_probability": proba, "is_fraud": (proba >= 0.5).astype(int)})


def main():
    data = prepare_data()
    comparison = compare_models(data)
    print(comparison.round(4))
    name, model, params, cv_score = tune_best_model(data, comparison)
    result = save_model(data, name, model, params, cv_score, comparison)
    print("Test result:", {k: round(v, 4) for k, v in result.items()})
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()