"""Behavior checks for the shared evaluation metrics on hand-computed cases."""

import json
import unittest

from src.evaluate import evaluate_predictions


class EvaluateTests(unittest.TestCase):
    def test_metrics_match_hand_computed_values(self):
        # tn=2, fp=1, fn=1, tp=2
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 0, 1, 1]
        scores = [0.1, 0.2, 0.7, 0.3, 0.8, 0.9]
        metrics = evaluate_predictions(y_true, y_pred, scores)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 1, 1, 2))
        self.assertAlmostEqual(metrics["precision_bot"], 2 / 3)
        self.assertAlmostEqual(metrics["recall_bot"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_bot"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        # 8 of the 9 bot/human pairs rank the bot higher.
        self.assertAlmostEqual(metrics["roc_auc"], 8 / 9)
        # Bots sit at ranks 1, 2 and 4: (1 + 1 + 3/4) / 3.
        self.assertAlmostEqual(metrics["ap"], (1 + 1 + 3 / 4) / 3)

    def test_no_predicted_bots_counts_as_zero_not_error(self):
        metrics = evaluate_predictions([0, 0, 1], [0, 0, 0], [0.0, 0.0, 0.0])
        self.assertEqual(metrics["precision_bot"], 0)
        self.assertEqual(metrics["f1_bot"], 0)
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertAlmostEqual(metrics["ap"], 1 / 3)

    def test_single_class_ranking_metrics_are_none_and_serializable(self):
        metrics = evaluate_predictions([0, 0], [0, 1], [0.1, 0.9])
        self.assertIsNone(metrics["ap"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 0, 0))
        json.dumps(metrics, allow_nan=False)

    def test_invalid_inputs_stop_processing(self):
        cases = {
            "length": ([0, 1], [0], [0.1, 0.2]),
            "empty": ([], [], []),
            "detector labels": ([0, 1], [-1, 1], [0.1, 0.2]),
            "text labels": (["human", "bot"], [0, 1], [0.1, 0.2]),
            "nan score": ([0, 1], [0, 1], [0.1, float("nan")]),
            "inf score": ([0, 1], [0, 1], [0.1, float("inf")]),
        }
        for name, (y_true, y_pred, scores) in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                evaluate_predictions(y_true, y_pred, scores)


if __name__ == "__main__":
    unittest.main()
