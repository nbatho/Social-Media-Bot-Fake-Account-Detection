"""Training entry points shared by the modeling notebooks.

Return fitted scikit-learn pipelines so preprocessing travels with each model.
Use identical feature columns and the shared data split across all models.
"""

from .config import RANDOM_STATE


def train_isolation_forest(X_train, *, random_state=RANDOM_STATE):
    """TODO: fit an Isolation Forest pipeline without passing labels to fit."""
    raise NotImplementedError("Chưa triển khai Isolation Forest.")


def train_one_class_svm(X_train):
    """TODO: fit an imputer/scaler/One-Class SVM pipeline without labels."""
    raise NotImplementedError("Chưa triển khai One-Class SVM.")


def train_dummy(X_train, y_train):
    """TODO: fit a most-frequent-class baseline on the shared training data."""
    raise NotImplementedError("Chưa triển khai Dummy baseline.")


def train_gradient_boosting(X_train, y_train, *, random_state=RANDOM_STATE):
    """TODO: fit a Gradient Boosting classification pipeline."""
    raise NotImplementedError("Chưa triển khai Gradient Boosting.")
