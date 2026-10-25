"""Shared evaluation contract: human=0, bot=1, larger scores indicate bots."""


def evaluate_predictions(y_true, y_pred, scores) -> dict:
    """TODO: calculate bot precision/recall/F1, macro-F1, AP, ROC-AUC, accuracy.

    Predictions use labels 0/1. Scores must be continuous: negative detector
    decision_function or classifier probability for class 1. Use zero_division=0
    where relevant and report undefined ranking metrics for single-class data.
    """
    raise NotImplementedError("Chưa triển khai hàm đánh giá dùng chung.")
