"""Reproduce notebook 03: train-only fits and validation-only model selection.

Run from the repository root with ``python -m src.supervised``. Test rows are
assigned by the shared splitter but are never scored or used for selection.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from .config import (
    FEATURE_COLUMNS, MODELS_DIR, PROJECT_ROOT, RANDOM_STATE, RAW_LABELS_PATH,
    REPORTS_DIR, SPLIT_FRACTIONS, SPLIT_PATH,
)
from .data import load_profiles, split_features
from .evaluate import evaluate_predictions
from .features import build_features
from .train import bot_probabilities, select_threshold, train_dummy, train_gradient_boosting


# Fixed before looking at validation. First candidate wins exact AP ties.
GRADIENT_BOOSTING_CANDIDATES = tuple(
    {"n_estimators": trees, "max_depth": depth, "learning_rate": 0.1,
     "min_samples_leaf": 5, "subsample": 1.0}
    for trees in (100, 200) for depth in (2, 3, 4)
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_supervised_model(
    model, path, *, thresholds, threshold, validation_metrics, fit_seconds,
    run_id, provenance,
) -> dict:
    """Persist the complete inference pipeline plus the notebook-04 handoff."""
    classifier = model.named_steps["classifier"]
    is_dummy = isinstance(classifier, DummyClassifier)
    bundle = {
        "schema_version": 1,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_mapping": {"human": 0, "bot": 1},
        "score_method": "predict_proba for classes_ == 1; larger means bot",
        "config": classifier.get_params(deep=False),
        "preprocessing": {"imputer": "median", "keep_empty_features": True},
        "thresholds": thresholds,
        "threshold": float(threshold),
        "threshold_comparison": ">=",
        "random_state": classifier.get_params().get("random_state", RANDOM_STATE),
        "fit_seconds": float(fit_seconds),
        "validation_metrics": validation_metrics,
        "sklearn_version": sklearn.__version__,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "run_id": run_id,
        "provenance": provenance,
        "selection": {"config": "fixed most_frequent baseline" if is_dummy else
                                "maximum validation AP; first candidate on ties",
                      "threshold": "fixed default; no threshold tuning" if is_dummy else
                                   "maximum validation bot F1; lowest cut on ties",
                      "refit_on_validation": False, "test_used": False},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def _report_metrics(metrics):
    return {**metrics, "confusion_matrix": json.dumps(metrics["confusion_matrix"])}


def run_supervised(*, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR) -> dict:
    """Train both baselines, save models, search results, metrics and provenance."""
    started = perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    profiles, _ = load_profiles()
    features = build_features(profiles)
    train, validation, _ = split_features(features)
    X_train, y_train = train[list(FEATURE_COLUMNS)], train["label"]
    X_val, y_val = validation[list(FEATURE_COLUMNS)], validation["label"].to_numpy()
    provenance = {
        "source_sha256": file_sha256(RAW_LABELS_PATH),
        "split_sha256": file_sha256(SPLIT_PATH),
        "split_fractions": SPLIT_FRACTIONS,
        "split_seed": RANDOM_STATE,
        "train_rows": len(train), "validation_rows": len(validation),
        "code_sha256": {f"src/{name}.py": file_sha256(PROJECT_ROOT / "src" / f"{name}.py")
                        for name in ("config", "data", "features", "train", "evaluate", "supervised")},
        "test_history": "Notebook 04 on importance-analysis records prior test use with interim models. "
                        "This run does not score test or use test for selection.",
    }
    models, fit_seconds = {}, {}
    start = perf_counter()
    models["dummy"] = train_dummy(X_train, y_train)
    fit_seconds["dummy"] = perf_counter() - start

    search_rows, best_ap = [], -np.inf
    for candidate_id, config in enumerate(GRADIENT_BOOSTING_CANDIDATES):
        start = perf_counter()
        candidate = train_gradient_boosting(X_train, y_train, **config)
        seconds = perf_counter() - start
        scores = bot_probabilities(candidate, X_val)
        metrics = evaluate_predictions(y_val, (scores >= 0.5).astype(int), scores)
        search_rows.append({"run_id": run_id, "split": "validation", "candidate_id": candidate_id,
                            **config, "ap": metrics["ap"], "roc_auc": metrics["roc_auc"],
                            "f1_bot_default": metrics["f1_bot"], "fit_seconds": seconds})
        if metrics["ap"] > best_ap:
            best_ap, selected_id = metrics["ap"], candidate_id
            models["gradient_boosting"] = candidate
            fit_seconds["gradient_boosting"] = seconds
    search = pd.DataFrame(search_rows)
    search["selected"] = search["candidate_id"].eq(selected_id)

    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows, prediction_frames, metadata = [], [], {}
    for name, model in models.items():
        start = perf_counter()
        scores = bot_probabilities(model, X_val)
        score_seconds = perf_counter() - start
        default = model.predict(X_val)
        thresholds = {"default": 0.5}
        threshold = 0.5
        if name == "gradient_boosting":
            threshold = select_threshold(y_val, scores)["threshold"]
            thresholds["validation_f1"] = threshold
        model_rows = []
        for policy, cut in thresholds.items():
            predictions = default if policy == "default" else (scores >= cut).astype(int)
            metrics = evaluate_predictions(y_val, predictions, scores)
            model_rows.append({
                "run_id": run_id, "model": name, "split": "validation",
                "threshold_policy": policy, "threshold": cut,
                **_report_metrics(metrics), "fit_seconds": fit_seconds[name],
                "score_seconds": score_seconds, "rows": len(validation),
                "model_path": f"models/{name}.joblib",
                "split_sha256": provenance["split_sha256"],
            })
        rows.extend(model_rows)
        selected_predictions = default if name == "dummy" else (scores >= threshold).astype(int)
        prediction_frames.append(pd.DataFrame({
            "run_id": run_id, "model": name, "user_id": validation["user_id"].to_numpy(),
            "label": y_val, "bot_score": scores, "prediction_default": default,
            "prediction_selected": selected_predictions, "selected_threshold": threshold,
        }))
        bundle = save_supervised_model(
            model, models_dir / f"{name}.joblib", thresholds=thresholds, threshold=threshold,
            validation_metrics=model_rows, fit_seconds=fit_seconds[name],
            run_id=run_id, provenance=provenance,
        )
        # Verify both probabilities and labels survive the actual artifact handoff.
        restored = joblib.load(models_dir / f"{name}.joblib")
        np.testing.assert_array_equal(bot_probabilities(restored["model"], X_val), scores)
        np.testing.assert_array_equal(restored["model"].predict(X_val), default)
        metadata[name] = {key: value for key, value in bundle.items() if key != "model"}

    results = pd.DataFrame(rows)
    results.to_csv(reports_dir / "supervised_validation_metrics.csv", index=False)
    search.to_csv(reports_dir / "supervised_validation_search.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        reports_dir / "supervised_validation_predictions.csv", index=False)
    manifest = {"run_id": run_id, "provenance": provenance, "models": metadata,
                "selected_candidate_id": selected_id,
                "total_seconds": perf_counter() - started}
    (reports_dir / "supervised_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"results": results, "search": search, "manifest": manifest}


if __name__ == "__main__":
    run = run_supervised()
    print(run["results"][["model", "threshold_policy", "threshold", "f1_bot", "ap", "roc_auc", "accuracy"]].to_string(index=False))
    print("Run:", run["manifest"]["run_id"])
