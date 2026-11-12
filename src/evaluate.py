"""Tính chỉ số, so sánh các mô hình và feature importance. Quy ước: human=0, bot=1."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from .config import FEATURE_COLUMNS, RANDOM_STATE
from .models import bot_scores, default_predictions


def evaluate_predictions(y_true, y_pred, scores) -> dict:
    """Precision/Recall/F1 lớp bot, Macro-F1, AP, ROC-AUC, Accuracy và confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision_bot": precision_score(y_true, y_pred, zero_division=0),
        "recall_bot": recall_score(y_true, y_pred, zero_division=0),
        "f1_bot": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0),
        "ap": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compare_models(bundles: dict, frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Mỗi mô hình 2 dòng: ngưỡng mặc định và ngưỡng đã chọn trên validation. Không fit lại."""
    y, rows = frame["label"], []
    for name, bundle in bundles.items():
        model, scores = bundle["model"], bot_scores(bundle["model"], frame)
        for setting, pred in (("default", default_predictions(model, frame)),
                              ("validation_f1", (scores >= bundle["threshold"]).astype(int))):
            rows.append({"split": split_name, "model": name, "setting": setting,
                         "threshold": None if setting == "default" else bundle["threshold"],
                         **evaluate_predictions(y, pred, scores)})
    return pd.DataFrame(rows)


def feature_importance(model, validation: pd.DataFrame) -> pd.DataFrame:
    """Impurity importance (học từ train) + permutation importance (AP giảm trên validation)."""
    perm = permutation_importance(model, validation[FEATURE_COLUMNS], validation["label"],
                                  scoring="average_precision", n_repeats=10, random_state=RANDOM_STATE)
    return pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "impurity_importance": model[-1].feature_importances_,
        "permutation_ap_drop": perm.importances_mean,
        "permutation_ap_drop_std": perm.importances_std,
    }).sort_values("permutation_ap_drop", ascending=False, ignore_index=True)
