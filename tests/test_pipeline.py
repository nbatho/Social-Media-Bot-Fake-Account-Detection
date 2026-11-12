"""Các kiểm tra chính: đặc trưng, chia tập, quy ước điểm, chọn ngưỡng, chỉ số."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.data import build_features, load_profiles, split_features
from src.evaluate import evaluate_predictions
from src.models import (bot_scores, default_predictions, select_threshold,
                        train_gradient_boosting, train_isolation_forest)


def write_csv(rows, folder):
    path = Path(folder) / "profiles.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def row(**overrides):
    base = {"id": "1", "account_type": "human", "followers_count": 10, "friends_count": 5,
            "statuses_count": 100, "account_age_days": 50, "description": "hello"}
    return {**base, **overrides}


def toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.lognormal(size=(n, 4)), columns=FEATURE_COLUMNS)
    frame["label"] = (frame["posting_frequency"] > frame["posting_frequency"].median()).astype(int)
    return frame


class FeatureTests(unittest.TestCase):
    def test_formulas_zero_denominators_and_bio(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv([row(), row(id="2", friends_count=0, account_age_days=0, description=""),
                              row(id="3", description="None")], folder)
            features = build_features(load_profiles(path))
        self.assertEqual(features["follower_following_ratio"].tolist(), [2.0, 10.0, 2.0])
        self.assertEqual(features["posting_frequency"].tolist(), [2.0, 100.0, 2.0])
        self.assertEqual(features["bio_length"].tolist(), [5, 0, 4])  # chữ "None" giữ nguyên
        self.assertEqual(features["user_id"].tolist(), ["1", "2", "3"])

    def test_negative_or_text_numbers_become_missing_not_clipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_csv([row(account_age_days=-3), row(id="2", followers_count="abc")], folder)
            features = build_features(load_profiles(path))
        self.assertTrue(features.loc[0, ["posting_frequency", "account_age_days"]].isna().all())
        self.assertTrue(pd.isna(features.loc[1, "follower_following_ratio"]))

    def test_bad_ids_labels_and_columns_stop(self):
        cases = {"duplicate id": [row(), row()], "empty id": [row(id="")],
                 "bad label": [row(account_type="robot")]}
        with tempfile.TemporaryDirectory() as folder:
            for name, rows in cases.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    load_profiles(write_csv(rows, folder))
            no_bio = pd.DataFrame([row()]).drop(columns="description")
            no_bio.to_csv(Path(folder) / "x.csv", index=False)
            with self.assertRaises(ValueError):
                load_profiles(Path(folder) / "x.csv")


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({"user_id": [str(i) for i in range(100)],
                                      "label": [1] * 30 + [0] * 70})

    def test_disjoint_complete_stratified_and_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            parts = split_features(self.features, path)
            self.assertEqual([len(p) for p in parts], [60, 20, 20])
            self.assertEqual(sum(p["label"].sum() for p in parts), 30)
            self.assertEqual(len(set().union(*(set(p["user_id"]) for p in parts))), 100)
            for part in parts:
                self.assertAlmostEqual(part["label"].mean(), 0.3)
            again = split_features(self.features.sample(frac=1, random_state=1), path)
            self.assertEqual(set(again[2]["user_id"]), set(parts[2]["user_id"]))

    def test_mismatched_saved_split_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            split_features(self.features, path)
            with self.assertRaises(ValueError):
                split_features(self.features.iloc[:50], path)


class ModelTests(unittest.TestCase):
    def test_detector_scores_extreme_account_as_bot(self):
        train = toy_frame()
        model = train_isolation_forest(train, n_estimators=50)
        extreme = pd.DataFrame([[1e7, 1e4, 1e5, 500]], columns=FEATURE_COLUMNS)
        self.assertGreater(bot_scores(model, extreme)[0], bot_scores(model, train).max() - 1e-9)
        self.assertEqual(default_predictions(model, extreme)[0], 1)

    def test_label_never_used_as_feature(self):
        train = toy_frame()
        shuffled = train.assign(label=train["label"].sample(frac=1, random_state=0).to_numpy())
        a, b = train_isolation_forest(train, 50), train_isolation_forest(shuffled, 50)
        np.testing.assert_array_equal(bot_scores(a, train), bot_scores(b, train))

    def test_gradient_boosting_learns_signal(self):
        train, validation = toy_frame(seed=0), toy_frame(seed=1)
        model = train_gradient_boosting(train, n_estimators=20, max_depth=2)
        scores = bot_scores(model, validation)
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertGreater(evaluate_predictions(validation["label"], scores >= 0.5, scores)["roc_auc"], 0.95)

    def test_select_threshold_maximizes_f1(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.6, 0.7, 0.65, 0.9])
        threshold = select_threshold(y, scores)
        best = max(evaluate_predictions(y, scores >= t, scores)["f1_bot"] for t in scores)
        self.assertEqual(evaluate_predictions(y, scores >= threshold, scores)["f1_bot"], best)


class MetricTests(unittest.TestCase):
    def test_metrics_match_hand_computed_values(self):
        m = evaluate_predictions([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1], [0.1, 0.2, 0.7, 0.3, 0.8, 0.9])
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (2, 1, 1, 2))
        self.assertAlmostEqual(m["f1_bot"], 2 / 3)
        self.assertAlmostEqual(m["roc_auc"], 8 / 9)
        self.assertAlmostEqual(m["ap"], (1 + 1 + 3 / 4) / 3)

    def test_larger_scores_must_mean_bot(self):
        y = [0, 0, 1, 1]
        self.assertEqual(evaluate_predictions(y, y, [0.1, 0.2, 0.8, 0.9])["roc_auc"], 1.0)
        self.assertEqual(evaluate_predictions(y, y, [0.9, 0.8, 0.2, 0.1])["roc_auc"], 0.0)

    def test_no_predicted_bots_is_zero_not_error(self):
        m = evaluate_predictions([0, 0, 1], [0, 0, 0], [0.0, 0.0, 0.0])
        self.assertEqual((m["precision_bot"], m["f1_bot"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
