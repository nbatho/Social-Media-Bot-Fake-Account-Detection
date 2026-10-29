"""Behavior checks for the shared, persisted train/validation/test split."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import SPLIT_FRACTIONS
from src.data import load_split, split_features


def feature_table(rows=1000, bot_share=0.3):
    bots = int(rows * bot_share)
    return pd.DataFrame({
        # Leading zeros check that IDs are never parsed as numbers.
        "user_id": pd.array([f"{i:019d}" for i in range(rows)], dtype="string"),
        "follower_following_ratio": pd.array(range(rows), dtype="Float64"),
        "label": [1] * bots + [0] * (rows - bots),
    })


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "split.csv"

    def tearDown(self):
        self.folder.cleanup()

    def test_users_are_disjoint_complete_and_stratified(self):
        features = feature_table()
        parts = split_features(features, self.path)
        ids = [set(part["user_id"]) for part in parts]
        self.assertEqual(set().union(*ids), set(features["user_id"]))
        self.assertEqual(sum(map(len, ids)), len(features))
        for name, part in zip(SPLIT_FRACTIONS, parts):
            with self.subTest(split=name):
                self.assertAlmostEqual(len(part) / len(features), SPLIT_FRACTIONS[name], delta=0.01)
                self.assertAlmostEqual(part["label"].mean(), 0.3, delta=0.01)
                self.assertEqual(list(part.columns), list(features.columns))

    def test_saved_split_is_reused_even_when_rows_are_reordered(self):
        features = feature_table()
        first = split_features(features, self.path)
        saved = self.path.read_bytes()
        shuffled = features.sample(frac=1, random_state=7)
        second = split_features(shuffled, self.path)
        self.assertEqual(self.path.read_bytes(), saved)
        for a, b in zip(first, second):
            self.assertEqual(set(a["user_id"]), set(b["user_id"]))
        self.assertEqual(load_split(self.path).index[0], "0000000000000000000")

    def test_mismatched_saved_split_stops_instead_of_overwriting(self):
        split_features(feature_table(), self.path)
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "does not match"):
            split_features(feature_table(rows=999), self.path)
        self.assertEqual(self.path.read_bytes(), saved)

    def test_invalid_inputs_stop_processing(self):
        features = feature_table()
        bad_inputs = {
            "missing label": (features.drop(columns="label"), "Missing columns"),
            "duplicate id": (features.assign(user_id="1"), "unique"),
            "missing id": (features.assign(user_id=pd.NA), "unique"),
            "text label": (features.assign(label="bot"), "label must use"),
        }
        for name, (bad, message) in bad_inputs.items():
            with self.subTest(input=name), self.assertRaisesRegex(ValueError, message):
                split_features(bad, self.path)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
