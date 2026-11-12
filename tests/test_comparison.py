"""Regression checks for the notebook-04 saved-model contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.comparison import (
    MODEL_NAMES, evaluate_models, file_sha256, load_bundles,
    model_predictions, model_scores, validate_validation_metrics,
)
from src.config import FEATURE_COLUMNS
from src.evaluate import evaluate_predictions
from src.train import train_gradient_boosting, train_isolation_forest, train_one_class_svm


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.split = self.root / "source.csv", self.root / "split.csv"
        self.source.write_text("source\n", encoding="utf-8")
        self.split.write_text("user_id,split\n001,train\n", encoding="utf-8")
        self.X = pd.DataFrame(np.random.default_rng(9).uniform(1, 10, (80, 4)), columns=FEATURE_COLUMNS)
        self.y = (self.X["posting_frequency"] > 7).astype(int)
        models = {
            "gradient_boosting": train_gradient_boosting(self.X, self.y, n_estimators=10),
            "isolation_forest": train_isolation_forest(self.X, n_estimators=10),
            "one_class_svm": train_one_class_svm(self.X),
        }
        for name, model in models.items():
            bundle = {
                "model": model, "feature_columns": list(FEATURE_COLUMNS),
                "sklearn_version": sklearn.__version__, "config": {}, "fit_seconds": 0.1,
                "run_id": "test-run", "thresholds": {"default": 0.5},
                "provenance": {"source_sha256": file_sha256(self.source), "split_sha256": file_sha256(self.split)},
            }
            scores = model_scores(name, bundle, self.X)
            bundle["thresholds"]["validation_f1"] = float(np.median(scores))
            bundle["validation_metrics"] = [evaluate_predictions(self.y, np.zeros(len(self.y)), scores)]
            joblib.dump(bundle, self.root / f"{name}.joblib")

    def load(self):
        return load_bundles(self.root, source_path=self.source, split_path=self.split)

    def test_three_required_models_roundtrip_without_refitting(self):
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("must not refit")):
            bundles = self.load()
            results = evaluate_models(bundles, self.X, self.y, "validation")
            validate_validation_metrics(bundles, results)
        self.assertEqual(set(results.model), {"isolation_forest", "one_class_svm", "gradient_boosting"})
        self.assertEqual(len(results), 6)
        for name in MODEL_NAMES:
            self.assertEqual(results.loc[results.model == name, "setting"].tolist(), ["default", "validation_f1"])
        for name in MODEL_NAMES:
            self.assertEqual(len(bundles[name]["model_sha256"]), 64)
            row = results[(results.model == name) & (results.setting == "default")].iloc[0]
            self.assertEqual(row.run_id, "test-run")
            self.assertEqual(row.tn + row.fp + row.fn + row.tp, len(self.y))
            if name == "gradient_boosting":
                expected = bundles[name]["model"].predict(self.X)
                self.assertAlmostEqual(row.accuracy, np.mean(expected == self.y))

    def test_source_and_split_mismatches_stop_before_evaluation(self):
        for field in ("source_sha256", "split_sha256"):
            with self.subTest(field=field):
                path = self.root / "isolation_forest.joblib"
                original = path.read_bytes()
                bundle = joblib.load(path)
                bundle["provenance"][field] = "wrong"
                joblib.dump(bundle, path)
                with self.assertRaisesRegex(ValueError, field):
                    self.load()
                path.write_bytes(original)

    def test_missing_file_schema_and_version_are_rejected(self):
        path = self.root / "isolation_forest.joblib"
        original = path.read_bytes()
        for field, value, message in (("feature_columns", [], "feature_columns"),
                                      ("sklearn_version", "old", "scikit-learn")):
            bundle = joblib.load(path)
            bundle[field] = value
            joblib.dump(bundle, path)
            with self.assertRaisesRegex(ValueError, message):
                self.load()
            path.write_bytes(original)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "isolation_forest.joblib"):
            self.load()

    def test_changed_validation_ap_is_rejected(self):
        bundles = self.load()
        results = evaluate_models(bundles, self.X, self.y, "validation")
        bundles["gradient_boosting"]["validation_metrics"][0]["ap"] = -1
        with self.assertRaisesRegex(ValueError, "gradient_boosting.*AP"):
            validate_validation_metrics(bundles, results)
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_validation_metrics(bundles, evaluate_models(bundles, self.X, self.y, "test"))

    def test_default_uses_classifier_tie_rule_instead_of_probability_cut(self):
        class TiedClassifier:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
        bundle = {"model": TiedClassifier()}
        scores = np.full(len(self.X), 0.5)
        default = model_predictions("gradient_boosting", bundle, self.X, None, scores)
        tuned = model_predictions("gradient_boosting", bundle, self.X, 0.5, scores)
        self.assertTrue((default == 0).all())
        self.assertTrue((tuned == 1).all())

    def test_invalid_threshold_and_split_are_rejected(self):
        bundles = self.load()
        bundles["gradient_boosting"]["thresholds"]["validation_f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_models(bundles, self.X, self.y, "test")
        with self.assertRaisesRegex(ValueError, "split"):
            evaluate_models(bundles, self.X, self.y, "train")


if __name__ == "__main__":
    unittest.main()
