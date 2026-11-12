"""Huấn luyện 3 mô hình, tìm cấu hình và ngưỡng trên validation, lưu/tải model.

Mọi hàm nhận cả bảng (train/validation/test) và tự lấy FEATURE_COLUMNS,
nên user_id và label không bao giờ lọt vào mô hình.
"""

from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM

from .config import FEATURE_COLUMNS, MODELS_DIR, RANDOM_STATE

DETECTORS = ("isolation_forest", "one_class_svm")


def train_isolation_forest(train, n_estimators=300):
    """Không dùng nhãn. Mô hình cây nên không cần log/scale."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1),
    )
    return model.fit(train[FEATURE_COLUMNS])


def train_one_class_svm(train, nu=0.1, gamma="scale"):
    """Không dùng nhãn. log1p + scale vì SVM đo khoảng cách và ratio lệch tới hàng triệu."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log1p),
        StandardScaler(),
        OneClassSVM(kernel="rbf", nu=nu, gamma=gamma),
    )
    return model.fit(train[FEATURE_COLUMNS])


def train_gradient_boosting(train, n_estimators=100, max_depth=3):
    """Có dùng nhãn."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_leaf=5, random_state=RANDOM_STATE),
    )
    return model.fit(train[FEATURE_COLUMNS], train["label"])


def is_detector(model):
    """Isolation Forest / One-Class SVM không có predict_proba."""
    return not hasattr(model, "predict_proba")


def bot_scores(model, frame):
    """Điểm liên tục, càng lớn càng giống bot (dùng cho AP, ROC-AUC, chọn ngưỡng)."""
    X = frame[FEATURE_COLUMNS]
    if is_detector(model):
        return -model.decision_function(X)   # detector: nhỏ = lạ → đảo dấu
    return model.predict_proba(X)[:, 1]      # classes_ luôn là [0, 1] → cột 1 là bot


def default_predictions(model, frame):
    """Quyết định mặc định: detector dùng predict() == -1 (bất thường), classifier dùng 0,5."""
    pred = model.predict(frame[FEATURE_COLUMNS])
    return (pred == -1).astype(int) if is_detector(model) else pred.astype(int)


def select_threshold(y_val, scores):
    """Ngưỡng cho F1 bot cao nhất. CHỈ gọi trên validation."""
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def search(train_fn, grid, train, validation):
    """Thử mọi tổ hợp trong grid, chọn theo AP validation (hòa thì giữ cấu hình đầu).

    Trả về (model tốt nhất, cấu hình tốt nhất, bảng kết quả).
    """
    rows, best = [], None
    for values in product(*grid.values()):
        config = dict(zip(grid, values))
        model = train_fn(train, **config)
        ap = average_precision_score(validation["label"], bot_scores(model, validation))
        rows.append({**config, "ap_validation": ap})
        if best is None or ap > best[2]:
            best = (model, config, ap)
    table = pd.DataFrame(rows).sort_values("ap_validation", ascending=False, ignore_index=True)
    return best[0], best[1], table


def save_model(name, model, config, threshold):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": config, "threshold": threshold,
                 "feature_columns": FEATURE_COLUMNS}, MODELS_DIR / f"{name}.joblib")


def load_model(name):
    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(f"{name}: model được train với bộ đặc trưng khác; hãy train lại.")
    return bundle
