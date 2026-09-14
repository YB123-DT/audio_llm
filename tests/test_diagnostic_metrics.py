import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_output_behavior import summarize_prediction_file
from scripts.analyze_forced_choice import _roc_auc
from scripts.analyze_slam_omni_diagnostics import _conditioned_parallelism
from scripts.analyze_decision_transfer import build_transfer_rows, summarize_transfer
from scripts.analyze_activation_patching import build_mechanism_rows, summarize_patch_rows
from scripts.run_activation_patching import _counterfactual_margin_effects, _pair_indices, _patch_hook

try:
    import torch as torch_lib
except ImportError:  # Local metric-only environments do not need torch.
    torch_lib = None


class DiagnosticMetricTests(unittest.TestCase):
    def test_counterfactual_effect_uses_both_candidate_margins(self) -> None:
        effects = _counterfactual_margin_effects(
            baseline_happy_happy_score=1.0,
            baseline_happy_sad_score=0.0,
            baseline_sad_happy_score=-1.0,
            baseline_sad_sad_score=0.0,
            patched_happy_target_happy_score=0.2,
            patched_happy_target_sad_score=0.1,
            patched_sad_target_happy_score=-0.1,
            patched_sad_target_sad_score=0.0,
        )
        self.assertAlmostEqual(effects["baseline_happy_margin"], 1.0)
        self.assertAlmostEqual(effects["baseline_sad_margin"], -1.0)
        self.assertAlmostEqual(effects["patched_happy_target_margin"], 0.1)
        self.assertAlmostEqual(effects["patched_sad_target_margin"], -0.1)
        self.assertAlmostEqual(effects["happy_effect_toward_sad"], 0.9)
        self.assertAlmostEqual(effects["sad_effect_toward_happy"], 0.9)
        self.assertAlmostEqual(effects["counterfactual_effect"], 0.9)

    def test_activation_patch_effect_and_mechanism_mapping(self) -> None:
        rows = []
        for patch_kind, patched_happy_margin, patched_sad_margin in (
            ("audio_tokens", 0.0, 0.0),
            ("decision_token", -0.2, 0.2),
        ):
            rows.append(
                {
                    "pair_id": "pair_0",
                    "layer_index": "7",
                    "layer": "layer_7",
                    "patch_kind": patch_kind,
                    "baseline_happy_margin": "1.0",
                    "baseline_sad_margin": "-1.0",
                    "patched_happy_target_margin": str(patched_happy_margin),
                    "patched_sad_target_margin": str(patched_sad_margin),
                    "error": "",
                }
            )
        summaries = summarize_patch_rows(rows)
        self.assertEqual(len(summaries), 2)
        self.assertAlmostEqual(summaries[0]["mean_counterfactual_effect"], 1.0)
        self.assertAlmostEqual(summaries[0]["sd_counterfactual_effect"], 0.0)
        self.assertAlmostEqual(summaries[0]["ci95_low_counterfactual_effect"], 1.0)
        self.assertAlmostEqual(summaries[0]["ci95_high_counterfactual_effect"], 1.0)
        mechanisms = build_mechanism_rows(summaries)
        self.assertEqual(len(mechanisms), 1)
        self.assertIn("downstream readout", mechanisms[0]["interpretation"])

    @unittest.skipIf(torch_lib is None, "torch is required for hook tensor checks")
    def test_activation_patch_pair_matching_and_hook_positions(self) -> None:
        rows = [
            {"pair_id": "p", "emotion": "sad", "actor": "01", "statement": "01", "repetition": "01", "intensity": "01", "sample_id": "p_sad"},
            {"pair_id": "p", "emotion": "happy", "actor": "01", "statement": "01", "repetition": "01", "intensity": "01", "sample_id": "p_happy"},
        ]
        pairs = _pair_indices(rows)
        self.assertEqual(pairs[0]["happy"]["sample_id"], "p_happy")
        donor = torch_lib.full((1, 2, 3), 9.0)
        output = (torch_lib.zeros((1, 5, 3)), "aux")
        patched = _patch_hook(donor, "audio_tokens", 1, 2, 4)(None, None, output)
        self.assertTrue(torch_lib.equal(patched[0][0, 0], torch_lib.zeros(3)))
        self.assertTrue(torch_lib.equal(patched[0][0, 1:3], donor[0]))
        self.assertEqual(patched[1], "aux")
    def test_decision_transfer_reports_probe_gap_and_layer_step(self) -> None:
        rows = []
        for family, layer_values in (
            ("llm_audio_mean", ((0.95, 0.90), (0.90, 0.85))),
            ("llm_decision", ((0.80, 0.70), (0.60, 0.50))),
        ):
            for layer_index, (speaker, statement) in enumerate(layer_values):
                rows.append(
                    {
                        "family": family,
                        "layer_index": str(layer_index),
                        "layer": f"layer_{layer_index}",
                        "speaker_held_out_accuracy": str(speaker),
                        "statement_held_out_mean": str(statement),
                    }
                )
        transfer = build_transfer_rows(rows)
        self.assertAlmostEqual(transfer[1]["speaker_gap_audio_minus_decision"], 0.30)
        self.assertAlmostEqual(transfer[1]["speaker_decision_step"], -0.20)
        self.assertAlmostEqual(transfer[1]["statement_gap_audio_minus_decision"], 0.35)
        summary = summarize_transfer(transfer)
        self.assertEqual(summary["largest_audio_minus_decision_gap"]["speaker"]["layer_index"], 1)
        self.assertEqual(summary["bottleneck_search_excludes_layer_indices"], [0])

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
