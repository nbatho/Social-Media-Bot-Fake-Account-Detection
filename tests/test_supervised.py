"""Supervised fitting, class-score convention, persistence and validation selection."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import FEATURE_COLUMNS
from src.supervised import run_supervised, save_supervised_model
from src.train import bot_probabilities, train_gradient_boosting


def training_data():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.uniform(1, 10, (80, 4)), columns=FEATURE_COLUMNS).astype("Float64")
    y = (X["posting_frequency"] > 7).astype(int)
    return X, y


class SupervisedTests(unittest.TestCase):
    def test_score_locates_bot_column_in_reversed_classes(self):
        class ReversedClasses:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                return np.tile([0.8, 0.2], (len(X), 1))

        X, _ = training_data()
        np.testing.assert_array_equal(bot_probabilities(ReversedClasses(), X), np.full(len(X), 0.8))

    def test_imputer_learns_only_train_even_after_validation_predictions(self):
        X, y = training_data()
        X.iloc[0, 0] = pd.NA
        X.iloc[1, 2] = pd.NA
        expected = X.median().to_numpy(dtype=float)
        validation = X.copy() * 10000
        validation.iloc[0] = pd.NA
        for train in (train_gradient_boosting,):
            with self.subTest(trainer=train.__name__):
                model = train(X, y)
                np.testing.assert_allclose(model.named_steps["imputer"].statistics_, expected)
                self.assertTrue(np.isfinite(bot_probabilities(model, validation)).all())
                np.testing.assert_allclose(model.named_steps["imputer"].statistics_, expected)

    def test_all_missing_training_feature_keeps_four_columns(self):
        X, y = training_data()
        X["bio_length"] = pd.NA
        model = train_gradient_boosting(X, y, n_estimators=10)
        self.assertEqual(model.named_steps["classifier"].n_features_in_, 4)
        self.assertTrue(np.isfinite(bot_probabilities(model, X)).all())

    def test_gradient_boosting_learns_signal_and_repeats_with_seed(self):
        X, y = training_data()
        first = train_gradient_boosting(X, y, random_state=17, subsample=0.8)
        second = train_gradient_boosting(X, y, random_state=17, subsample=0.8)
        np.testing.assert_array_equal(bot_probabilities(first, X), bot_probabilities(second, X))
        self.assertGreater(f1_score(y, first.predict(X)), 0.95)

    def test_schema_and_labels_are_checked(self):
        X, y = training_data()
        bad_inputs = [X.assign(label=y), X.assign(user_id="1"), X[list(reversed(FEATURE_COLUMNS))]]
        for train in (train_gradient_boosting,):
            for bad in bad_inputs:
                with self.subTest(trainer=train.__name__), self.assertRaises(ValueError):
                    train(bad, y)
            for bad_y in (y.iloc[::-1], np.full(len(X), 2), y.iloc[:-1], np.ones((len(X), 1))):
                with self.subTest(trainer=train.__name__), self.assertRaises(ValueError):
                    train(X, bad_y)
            model = train(X, y)
            with self.assertRaisesRegex(ValueError, "Expected DataFrame columns"):
                model.predict(X.assign(label=y))

    def test_bundle_roundtrip_preserves_probabilities_predictions_and_metadata(self):
        X, y = training_data()
        for train in (train_gradient_boosting,):
            with self.subTest(trainer=train.__name__), TemporaryDirectory() as directory:
                model = train(X, y)
                path = Path(directory) / "model.joblib"
                bundle = save_supervised_model(
                    model, path, thresholds={"default": 0.5}, threshold=0.5,
                    validation_metrics=[], fit_seconds=0.1, run_id="test-run", provenance={"split_sha256": "abc"},
                )
                restored = joblib.load(path)
                np.testing.assert_array_equal(bot_probabilities(restored["model"], X), bot_probabilities(model, X))
                np.testing.assert_array_equal(restored["model"].predict(X), model.predict(X))
                for key in ("config", "feature_columns", "thresholds", "random_state", "versions",
                            "fit_seconds", "run_id", "provenance"):
                    self.assertEqual(restored[key], bundle[key])
                self.assertEqual(restored["selection"]["config"], "maximum validation AP; first candidate on ties")
                self.assertEqual(restored["selection"]["threshold"], "maximum validation bot F1; lowest cut on ties")
                with self.assertRaises(ValueError):
                    restored["model"].predict(X[list(reversed(FEATURE_COLUMNS))])

    def test_runner_selects_on_validation_and_never_uses_test(self):
        X, y = training_data()
        features = X.assign(user_id=[str(i) for i in range(len(X))], label=y)
        train, validation = features.iloc[:60], features.iloc[60:]
        candidates = ({"n_estimators": 5, "max_depth": 1}, {"n_estimators": 10, "max_depth": 2})
        with TemporaryDirectory() as directory, patch("src.supervised.load_profiles", return_value=(None, {})), \
                patch("src.supervised.build_features", return_value=features), \
                patch("src.supervised.split_features", return_value=(train, validation, object())), \
                patch("src.supervised.file_sha256", return_value="abc"), \
                patch("src.supervised.GRADIENT_BOOSTING_CANDIDATES", candidates):
            root = Path(directory)
            run = run_supervised(models_dir=root / "models", reports_dir=root / "reports")
            search = run["search"]
            self.assertEqual(run["manifest"]["selected_candidate_id"], int(search["ap"].idxmax()))
            self.assertEqual(set(run["results"]["split"]), {"validation"})
            bundle = joblib.load(root / "models/gradient_boosting.joblib")
            self.assertFalse(bundle["selection"]["test_used"])
            scores = bot_probabilities(bundle["model"], validation[list(FEATURE_COLUMNS)])
            selected_f1 = f1_score(validation["label"], scores >= bundle["threshold"])
            self.assertAlmostEqual(selected_f1, max(f1_score(validation["label"], scores >= t) for t in np.unique(scores)))
            np.testing.assert_allclose(bundle["model"].named_steps["imputer"].statistics_, train[list(FEATURE_COLUMNS)].median())
            manifest = root / "reports/supervised_run.json"
            self.assertTrue(manifest.exists())
            self.assertEqual(set(run["manifest"]["models"]), {"gradient_boosting"})
            self.assertEqual(set(run["results"]["model"]), {"gradient_boosting"})
            self.assertEqual(len(pd.read_csv(root / "reports/supervised_validation_predictions.csv")), len(validation))


if __name__ == "__main__":
    unittest.main()
