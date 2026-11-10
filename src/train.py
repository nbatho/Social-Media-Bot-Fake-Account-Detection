"""Training entry points shared by the modeling notebooks.

Return fitted scikit-learn pipelines so preprocessing travels with each model.
Use identical feature columns and the shared data split across all models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM

from .config import FEATURE_COLUMNS, RANDOM_STATE


def feature_matrix(X) -> pd.DataFrame:
    """Return the four model features as float64; reject IDs, labels or reordering."""
    if not isinstance(X, pd.DataFrame) or list(X.columns) != list(FEATURE_COLUMNS):
        columns = list(X.columns) if isinstance(X, pd.DataFrame) else type(X).__name__
        raise ValueError(f"Expected DataFrame columns {list(FEATURE_COLUMNS)}, got {columns}.")
    # Nullable Float64 missing values become NaN for the imputer.
    return X.astype("float64")


def train_isolation_forest(
    X_train, *, n_estimators=300, contamination="auto", random_state=RANDOM_STATE
) -> Pipeline:
    """Fit imputer + Isolation Forest on X_train only; labels never reach fit.

    Trees split on thresholds, so monotone transforms or scaling are unnecessary.
    """
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=random_state, n_jobs=-1,
        )),
    ])
    return model.fit(feature_matrix(X_train))


def train_one_class_svm(X_train, *, nu=0.1, gamma="scale") -> Pipeline:
    """Fit imputer + log1p + scaler + RBF One-Class SVM on X_train without labels.

    Features are nonnegative and heavy-tailed (ratios reach millions), so log1p
    runs before scaling; otherwise a few extreme accounts dominate the kernel.
    """
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
        ("detector", OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)),
    ])
    return model.fit(feature_matrix(X_train))


def anomaly_scores(model: Pipeline, X) -> np.ndarray:
    """Continuous bot score: negative decision_function, larger means more anomalous."""
    return -model.decision_function(feature_matrix(X))


def predict_bot(model: Pipeline, X, threshold: float | None = None) -> np.ndarray:
    """Return human=0/bot=1. None uses the detector's own cut (-1 outlier -> bot)."""
    if threshold is None:
        return (model.predict(feature_matrix(X)) == -1).astype("int64")
    return (anomaly_scores(model, X) >= threshold).astype("int64")


def select_threshold(y_val, scores) -> dict:
    """Pick the score cut maximizing bot F1 on validation labels (never on test).

    predicted_bot_share near 1 means the best cut is "everything is a bot":
    the scores do not separate the classes better than the trivial rule.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, scores, pos_label=1)
    # The last precision/recall pair has no threshold.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[best]), "f1": float(f1[best]),
        "precision": float(precision[best]), "recall": float(recall[best]),
        "predicted_bot_share": float(np.mean(np.asarray(scores) >= thresholds[best])),
    }


def train_dummy(X_train, y_train):
    """TODO: fit a most-frequent-class baseline on the shared training data."""
    raise NotImplementedError("Chưa triển khai Dummy baseline.")


def train_gradient_boosting(X_train, y_train, *, random_state=RANDOM_STATE):
    """TODO: fit a Gradient Boosting classification pipeline."""
    raise NotImplementedError("Chưa triển khai Gradient Boosting.")
