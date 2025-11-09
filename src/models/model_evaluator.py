"""Model evaluation utilities for phishing vulnerability models.

Saves metrics and plots into models/evaluation/ by default.
"""
from typing import Dict
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
)


def evaluate_and_save(y_true, y_pred, y_proba, out_dir: str = "models/evaluation") -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = float(roc_auc_score(y_true, y_proba))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # Plots
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"))
    plt.close()

    metrics = {
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(os.path.join(out_dir, "performance_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics