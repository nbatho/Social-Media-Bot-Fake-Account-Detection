"""Validate saved-model handoffs and compare fixed models without retraining."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import FEATURE_COLUMNS, MODELS_DIR, RAW_LABELS_PATH, SPLIT_PATH
from .evaluate import evaluate_predictions
from .train import anomaly_scores, bot_probabilities, feature_matrix, predict_bot


DETECTORS = ("isolation_forest", "one_class_svm")
MODEL_NAMES = ("dummy", *DETECTORS, "gradient_boosting")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundles(models_dir=MODELS_DIR, *, source_path=RAW_LABELS_PATH, split_path=SPLIT_PATH):
    """Reject stale models from a different dataset, split, schema or runtime."""
    expected = {"source_sha256": file_sha256(source_path), "split_sha256": file_sha256(split_path)}
    bundles = {}
    for name in MODEL_NAMES:
        path = Path(models_dir) / f"{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}; run notebooks 02 and 03 first.")
        bundle = joblib.load(path)
        if bundle.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ValueError(f"{name}: feature_columns do not match FEATURE_COLUMNS.")
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError(f"{name}: scikit-learn version differs; regenerate the model.")
        for key, value in expected.items():
            if bundle.get("provenance", {}).get(key) != value:
                raise ValueError(f"{name}: {key} mismatch; regenerate from the shared source/split.")
        if not bundle.get("run_id"):
            raise ValueError(f"{name}: missing run_id; regenerate the model.")
        bundle["model_sha256"] = file_sha256(path)
        bundles[name] = bundle
    return bundles


def model_scores(name, bundle, X):
    return (anomaly_scores(bundle["model"], X) if name in DETECTORS
            else bot_probabilities(bundle["model"], X))


def model_predictions(name, bundle, X, threshold, scores):
    if name in DETECTORS:
        return predict_bot(bundle["model"], X, threshold)
    if threshold is None:
        # Preserve the classifier's tie rule and Dummy's most-frequent baseline.
        return bundle["model"].predict(feature_matrix(X))
    return (scores >= threshold).astype("int64")


def evaluate_models(bundles, X, y, split_name):
    """Use default predictions and existing validation thresholds; never fit."""
    if split_name not in ("validation", "test"):
        raise ValueError("Comparison split must be validation or test.")
    evaluated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for name in MODEL_NAMES:
        bundle = bundles[name]
        start = perf_counter()
        scores = model_scores(name, bundle, X)
        seconds = perf_counter() - start
        cuts = {"default": None}
        selected_cut = bundle["thresholds"].get("validation_f1")
        if name != "dummy" and selected_cut is not None:
            if not np.isfinite(selected_cut):
                raise ValueError(f"{name}: validation threshold must be finite.")
            cuts["validation_f1"] = selected_cut
        for setting, cut in cuts.items():
            pred = model_predictions(name, bundle, X, cut, scores)
            metrics = evaluate_predictions(y, pred, scores)
            metrics["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
            rows.append({
                "evaluation_split": split_name, "model": name, "setting": setting,
                "threshold": cut, **metrics,
                "fit_seconds": bundle["fit_seconds"], "score_seconds": seconds,
                "model_file": f"models/{name}.joblib",
                "config": json.dumps(bundle["config"], sort_keys=True),
                "sklearn_version": bundle["sklearn_version"],
                "run_id": bundle["run_id"], "model_sha256": bundle["model_sha256"],
                "source_sha256": bundle["provenance"]["source_sha256"],
                "split_sha256": bundle["provenance"]["split_sha256"],
                "evaluated_at_utc": evaluated_at,
            })
    return pd.DataFrame(rows)


def validate_validation_metrics(bundles, results):
    """Check the saved validation AP for every model before looking at test."""
    if not results["evaluation_split"].eq("validation").all():
        raise ValueError("Expected validation results for the handoff check.")
    for name in MODEL_NAMES:
        saved = bundles[name]["validation_metrics"][0]["ap"]
        current = results.loc[results["model"] == name, "ap"].iloc[0]
        if not np.isclose(saved, current, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name}: validation AP differs from the saved training run.")
