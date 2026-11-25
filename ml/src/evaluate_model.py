# evaluate_model.py - tests the saved model on the TEST data
# Run:  python ml/src/evaluate_model.py

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SRC_DIR))
sys.path.append(SRC_DIR)
import feature_engineering as fe
import predict

FIGURE_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "evaluation_report.json")


def get_test_scores():
    """Return y_test and the fraud probability of every test row."""
    X_train, X_test, y_train, y_test = fe.load_train_test()
    proba = predict.load_model().predict_proba(X_test)[:, 1]
    return y_test.values, proba


def get_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def save_plots(y_true, proba, threshold):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # ROC and Precision-Recall curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, label="model")
    axes[0].plot([0, 1], [0, 1], "--", label="random")
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(recall, precision, label="model")
    axes[1].axhline(y_true.mean(), color="red", linestyle="--", label="random")
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "20_roc_pr_curves.png"), dpi=120)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "21_confusion_matrix.png"), dpi=120)
    plt.close()


def main():
    threshold = predict.load_info()["threshold"]
    y_true, proba = get_test_scores()

    metrics = get_metrics(y_true, proba, threshold)
    save_plots(y_true, proba, threshold)

    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test rows:", len(y_true), " Fraud rows:", int(y_true.sum()))
    for key, value in metrics.items():
        print(f"{key:16s} {value}")
    print()
    print(classification_report(y_true, (proba >= threshold).astype(int),
                                target_names=["Legit", "Fraud"], zero_division=0))
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
    