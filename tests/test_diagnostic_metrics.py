import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_output_behavior import summarize_prediction_file
from scripts.analyze_forced_choice import _roc_auc
from scripts.analyze_slam_omni_diagnostics import _conditioned_parallelism


class DiagnosticMetricTests(unittest.TestCase):
    def test_roc_auc_uses_pairwise_tie_handling(self) -> None:
        self.assertAlmostEqual(_roc_auc([1, 0], [0.9, 0.1]), 1.0)
        self.assertAlmostEqual(_roc_auc([1, 0], [0.5, 0.5]), 0.5)

    def test_conditioned_parallelism_controls_the_requested_factors(self) -> None:
        specs = [
            ("pair_00", "01", "01", np.array([1.0, 0.0])),
            ("pair_01", "01", "02", np.array([1.0, 0.0])),
            ("pair_02", "02", "01", np.array([0.0, 1.0])),
            ("pair_03", "02", "02", np.array([0.0, 1.0])),
            ("pair_04", "01", "01", np.array([1.0, 0.0])),
            ("pair_05", "01", "02", np.array([0.0, 1.0])),
            ("pair_06", "02", "01", np.array([1.0, 0.0])),
            ("pair_07", "02", "02", np.array([-1.0, 0.0])),
        ]
        metadata = []
        features = []
        for pair_id, repetition, statement, delta in specs:
            for emotion, vector in (("happy", np.zeros(2)), ("sad", delta)):
                metadata.append(
                    {
                        "pair_id": pair_id,
                        "actor": "01" if int(pair_id[-1]) < 4 else "02",
                        "statement": statement,
                        "repetition": repetition,
                        "intensity": "01",
                        "emotion": emotion,
                    }
                )
                features.append(vector)

        metrics = _conditioned_parallelism(np.asarray(features), metadata)
        for scope in ("all", "text", "speaker", "repetition"):
            self.assertEqual(metrics[scope]["groups"], 4 if scope != "all" else 1)
            self.assertEqual(metrics[scope]["comparisons"], 4 if scope != "all" else 28)
        self.assertAlmostEqual(metrics["text"]["parallelism_cosine"], 0.25)
        self.assertAlmostEqual(metrics["speaker"]["parallelism_cosine"], 0.25)
        self.assertAlmostEqual(metrics["repetition"]["parallelism_cosine"], 0.25)

    def test_output_sensitivity_counts_exact_emotion_swap_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            fieldnames = ["sample_id", "pair_id", "actor", "statement", "repetition", "intensity", "emotion", "model_output"]
            rows = [
                {"sample_id": "pair_0_happy", "pair_id": "pair_0", "actor": "01", "statement": "01", "repetition": "01", "intensity": "01", "emotion": "happy", "model_output": "same"},
                {"sample_id": "pair_0_sad", "pair_id": "pair_0", "actor": "01", "statement": "01", "repetition": "01", "intensity": "01", "emotion": "sad", "model_output": "same"},
                {"sample_id": "pair_1_happy", "pair_id": "pair_1", "actor": "01", "statement": "02", "repetition": "01", "intensity": "01", "emotion": "happy", "model_output": "happy text"},
                {"sample_id": "pair_1_sad", "pair_id": "pair_1", "actor": "01", "statement": "02", "repetition": "01", "intensity": "01", "emotion": "sad", "model_output": "sad text"},
            ]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            summary, pairs = summarize_prediction_file("max8", path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(summary["same_output_pairs"], 1)
            self.assertEqual(summary["changed_output_pairs"], 1)
            self.assertAlmostEqual(summary["output_sensitivity"], 0.5)
            self.assertEqual(summary["statement_stats"]["01"]["mode_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
