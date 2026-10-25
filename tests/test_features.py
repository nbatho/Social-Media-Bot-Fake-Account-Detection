"""Behavior checks for identity validation, missing data and feature formulas."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, FEATURE_OUTPUT_COLUMNS
from src.data import load_profiles, normalize_profiles
from src.features import build_features, run_pipeline, summarize_features


def source_row(**changes):
    row = {
        "id": "9876543210987654321", "account_type": "human",
        "followers_count": "10", "friends_count": "2", "statuses_count": "12",
        "created_at": "2020-01-01 00:00:00", "description": "hello",
        "account_age_days": "3", "average_tweets_per_day": "4.000",
    }
    return {**row, **changes}


class FeatureTests(unittest.TestCase):
    def extract(self, **changes):
        profiles, quality = normalize_profiles(pd.DataFrame([source_row(**changes)]))
        return profiles, quality, build_features(profiles)

    def test_formula_order_long_id_and_input_not_mutated(self):
        raw = pd.DataFrame([source_row()])
        before = raw.copy(deep=True)
        profiles, _ = normalize_profiles(raw)
        features = build_features(profiles)
        pd.testing.assert_frame_equal(raw, before)
        self.assertEqual(list(features.columns), list(FEATURE_OUTPUT_COLUMNS))
        self.assertEqual(features.loc[0, "user_id"], "9876543210987654321")
        self.assertEqual(features.loc[0, "follower_following_ratio"], 5)
        self.assertEqual(features.loc[0, "posting_frequency"], 4)
        self.assertEqual(features.loc[0, "account_age_days"], 3)
        self.assertEqual(features.loc[0, "label"], 0)

    def test_zero_following_and_age_are_valid(self):
        _, quality, features = self.extract(friends_count="0", account_age_days="0")
        self.assertEqual(features.loc[0, "follower_following_ratio"], 10)
        self.assertEqual(features.loc[0, "posting_frequency"], 12)
        self.assertEqual(features.loc[0, "account_age_days"], 0)
        self.assertEqual(quality["numeric_columns"]["following_count"]["valid_zero"], 1)

    def test_negative_age_is_missing_not_clipped(self):
        _, quality, features = self.extract(account_age_days="-2")
        self.assertTrue(pd.isna(features.loc[0, "account_age_days"]))
        self.assertTrue(pd.isna(features.loc[0, "posting_frequency"]))
        self.assertEqual(quality["numeric_columns"]["account_age_days"]["negative"], 1)
        self.assertEqual(quality["rows_removed"], 0)

    def test_invalid_numeric_categories_and_affected_features(self):
        cases = [("", "missing"), (None, "missing"), (np.nan, "missing"),
                 ("NaN", "missing"), ("oops", "non_numeric"),
                 ("inf", "non_finite"), ("-inf", "non_finite"),
                 ("-1", "negative"), ("1.5", "fractional")]
        for value, category in cases:
            for raw_column, normalized, feature in (
                ("followers_count", "followers_count", "follower_following_ratio"),
                ("friends_count", "following_count", "follower_following_ratio"),
                ("statuses_count", "statuses_count", "posting_frequency"),
            ):
                with self.subTest(value=value, column=raw_column):
                    _, quality, features = self.extract(**{raw_column: value})
                    self.assertTrue(pd.isna(features.loc[0, feature]))
                    self.assertEqual(quality["numeric_columns"][normalized][category], 1)
                    self.assertEqual(len(features), 1)
                    self.assertFalse(np.isinf(features[list(FEATURE_COLUMNS)].to_numpy(dtype=float, na_value=np.nan)).any())

    def test_age_missing_invalid_and_fractional(self):
        for value in ("", None, "not a number", "inf"):
            with self.subTest(value=value):
                _, _, features = self.extract(account_age_days=value)
                self.assertTrue(pd.isna(features.loc[0, "account_age_days"]))
                self.assertTrue(pd.isna(features.loc[0, "posting_frequency"]))
        _, _, features = self.extract(account_age_days="2.5")
        self.assertEqual(features.loc[0, "posting_frequency"], 12 / 2.5)

    def test_null_empty_unicode_newlines_and_literal_bios(self):
        for bio in (None, pd.NA, np.nan, "", "None", "nan", "NA", "Tiếng Việt 🤖\nxin chào", " \n"):
            with self.subTest(bio=bio):
                _, _, features = self.extract(description=bio)
                expected = 0 if pd.isna(bio) else len(bio)
                self.assertEqual(features.loc[0, "bio_length"], expected)

    def test_missing_id_and_invalid_labels_stop_processing(self):
        for value in (None, "", "  "):
            with self.subTest(id=value), self.assertRaisesRegex(ValueError, "missing_ids"):
                self.extract(id=value)
        for value in ("robot", None, "", "Bot", " human "):
            with self.subTest(label=value), self.assertRaisesRegex(ValueError, "invalid_labels"):
                self.extract(account_type=value)

    def test_duplicate_and_conflicting_ids_stop_processing(self):
        for second_label, conflict_count in (("human", 0), ("bot", 1)):
            with self.subTest(second_label=second_label):
                raw = pd.DataFrame([source_row(), source_row(account_type=second_label)])
                with self.assertRaises(ValueError) as error:
                    normalize_profiles(raw)
                self.assertIn('"duplicate_ids": 1', str(error.exception))
                self.assertIn(f'"conflicting_ids": {conflict_count}', str(error.exception))

    def test_missing_schema_and_unverified_index_stop_processing(self):
        raw = pd.DataFrame([source_row()])
        with self.assertRaisesRegex(ValueError, "account_age_days"):
            normalize_profiles(raw.drop(columns="account_age_days"))
        raw["Unnamed: 0"] = "99"
        with self.assertRaisesRegex(ValueError, "Unverified source index"):
            normalize_profiles(raw)
        raw["Unnamed: 0"] = "0"
        profiles, quality = normalize_profiles(raw)
        self.assertNotIn("Unnamed: 0", profiles.columns)
        self.assertEqual(quality["dropped_source_index_columns"], ["Unnamed: 0"])
        self.assertNotIn("observed_at", profiles.columns)

    def test_rounding_comparison_reports_differences_without_rejecting(self):
        rows = [source_row(id="1", statuses_count="1", account_age_days="3", average_tweets_per_day="0.333"),
                source_row(id="2", average_tweets_per_day="99"),
                source_row(id="3", average_tweets_per_day="")]
        profiles, quality = normalize_profiles(pd.DataFrame(rows))
        features = build_features(profiles)
        summary = summarize_features(profiles, features, quality)
        comparison = summary["posting_frequency_comparison"]
        self.assertEqual(comparison["within_rounding_tolerance"], 1)
        self.assertEqual(comparison["beyond_rounding_tolerance"], 1)
        self.assertEqual(comparison["uncomparable_rows"], 1)
        self.assertEqual(len(features), 3)

    def test_no_comparable_source_frequency_serializes_as_null(self):
        profiles, quality, features = self.extract(average_tweets_per_day="")
        summary = summarize_features(profiles, features, quality)
        self.assertIsNone(summary["posting_frequency_comparison"]["max_absolute_difference"])
        json.dumps(summary, allow_nan=False)

    def test_csv_pipeline_preserves_long_ids_literal_bios_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, report = root / "raw.csv", root / "features.csv", root / "summary.json"
            raw = pd.DataFrame([source_row(description="None"), source_row(id="0000000000000000002", description="")])
            raw.to_csv(source, index=True, encoding="utf-8")
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            profiles, _ = load_profiles(source)
            self.assertEqual(profiles.loc[0, "description"], "None")
            features, summary = run_pipeline(source, output, report)
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            saved = pd.read_csv(output, dtype={"user_id": "string"})
            self.assertEqual(saved["user_id"].tolist(), raw["id"].tolist())
            self.assertEqual(saved["bio_length"].tolist(), [4, 0])
            self.assertTrue(summary["source"]["unchanged"])
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["output"]["rows"], 2)
            first_csv, first_json = output.read_bytes(), report.read_bytes()
            run_pipeline(source, output, report)
            self.assertEqual(output.read_bytes(), first_csv)
            self.assertEqual(report.read_bytes(), first_json)
            self.assertEqual(features.shape, (2, 6))

    def test_pipeline_refuses_source_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "raw.csv"
            pd.DataFrame([source_row()]).to_csv(source, index=False)
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                run_pipeline(source, source, Path(folder) / "summary.json")
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
