"""Shared evaluation contract: human=0, bot=1, larger scores indicate bots."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, scores) -> dict:
    """Calculate bot precision/recall/F1, macro-F1, AP, ROC-AUC, accuracy and counts.

    Predictions use labels 0/1. Scores must be continuous: negative detector
    decision_function or classifier probability for class 1. Undefined precision
    or recall count as 0; AP and ROC-AUC are None for single-class data.
    """
    y_true, y_pred, scores = np.asarray(y_true), np.asarray(y_pred), np.asarray(scores, dtype=float)
    if (any(values.ndim != 1 for values in (y_true, y_pred, scores))
            or not len(y_true) == len(y_pred) == len(scores) or len(y_true) == 0):
        raise ValueError("y_true, y_pred and scores must be non-empty and the same length.")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must contain only human=0 and bot=1.")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(y_true)) < 2
    return {
        "precision_bot": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_bot": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_bot": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "ap": None if single_class else float(average_precision_score(y_true, scores)),
        "roc_auc": None if single_class else float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "flagged_share": float(np.mean(y_pred)),
    }
