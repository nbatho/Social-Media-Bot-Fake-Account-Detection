"""Behavior checks for the unsupervised detectors and their score conventions."""

import unittest

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.train import (
    anomaly_scores, predict_bot, select_threshold,
    train_isolation_forest, train_one_class_svm,
)


def feature_frame(rows=300, seed=0):
    """Typical profiles plus one extreme, bot-like account in the last row."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "follower_following_ratio": rng.lognormal(0.5, 0.5, rows),
        "posting_frequency": rng.lognormal(1.0, 0.5, rows),
        "account_age_days": rng.normal(3000, 300, rows).clip(min=1),
        "bio_length": rng.integers(40, 120, rows).astype(float),
    }).astype("Float64")
    frame.iloc[-1] = [5e6, 900.0, 500.0, 0.0]
    return frame


TRAINERS = {
    "isolation_forest": lambda X: train_isolation_forest(X, n_estimators=50),
    "one_class_svm": train_one_class_svm,
}


class DetectorTests(unittest.TestCase):
    def test_extreme_account_scores_highest_and_predicts_bot(self):
        X = feature_frame()
        for name, train in TRAINERS.items():
            with self.subTest(model=name):
                model = train(X)
                scores = anomaly_scores(model, X)
                self.assertEqual(int(np.argmax(scores)), len(X) - 1)
                predictions = predict_bot(model, X)
                self.assertEqual(set(np.unique(predictions)) - {0, 1}, set())
                self.assertEqual(predictions[-1], 1)

    def test_default_cut_matches_zero_score_threshold(self):
        X = feature_frame()
        for name, train in TRAINERS.items():
            with self.subTest(model=name):
                model = train(X)
                scores = anomaly_scores(model, X)
                np.testing.assert_array_equal(predict_bot(model, X), (scores > 0).astype("int64"))

    def test_missing_values_are_imputed_from_training_data(self):
        X = feature_frame()
        X.iloc[0, 0] = pd.NA
        X.iloc[1, 3] = pd.NA
        for name, train in TRAINERS.items():
            with self.subTest(model=name):
                scores = anomaly_scores(train(X), X)
                self.assertTrue(np.isfinite(scores).all())

    def test_ids_labels_and_column_order_are_rejected(self):
        X = feature_frame()
        bad_inputs = {
            "label": X.assign(label=0),
            "user_id": X.assign(user_id="1"),
            "reordered": X[list(reversed(FEATURE_COLUMNS))],
            "array": X.to_numpy(dtype=float),
        }
        for name, bad in bad_inputs.items():
            with self.subTest(input=name), self.assertRaisesRegex(ValueError, "Expected DataFrame columns"):
                train_one_class_svm(bad)

    def test_isolation_forest_is_reproducible_with_seed(self):
        X = feature_frame()
        first = anomaly_scores(train_isolation_forest(X, n_estimators=50), X)
        second = anomaly_scores(train_isolation_forest(X, n_estimators=50), X)
        np.testing.assert_array_equal(first, second)

    def test_select_threshold_maximizes_validation_f1(self):
        y_val = np.array([0, 0, 0, 1, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        choice = select_threshold(y_val, scores)
        self.assertEqual(choice["threshold"], 0.4)
        self.assertAlmostEqual(choice["f1"], 6 / 7)
        self.assertAlmostEqual(choice["precision"], 0.75)
        self.assertEqual(choice["recall"], 1.0)
        self.assertAlmostEqual(choice["predicted_bot_share"], 4 / 7)


if __name__ == "__main__":
    unittest.main()
