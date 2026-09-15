#!/usr/bin/env python3
"""Run layer17 audio donor controls under a single-token forced choice.

The output keeps one row per matched happy/sad pair.  Controls differ only in
which frozen layer17 audio state is donated to each target:

``matched_swap``
    The strict happy/sad counterpart (reference condition).
``same_emotion``
    A donor with the same emotion and the same statement/repetition/intensity,
    but a different actor.
``random_donor``
    A deterministic donor sampled from all other manifest rows.

Effects are oriented toward the donor emotion, so same-emotion and random
donors can be compared with the matched emotion swap without changing the
forced-choice scoring rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.run_activation_patching import (
        DEFAULT_SEED,
        _capture_all_donors,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
        _score_all_baselines,
        _score_direction,
        _self_patch_check,
    )
    from scripts.slam_omni_diagnostics import FORCED_CHOICE_CONDITIONS, load_model, read_rows
except ModuleNotFoundError:  # Direct ``python scripts/run_donor_control_patching.py`` entry point.
    from run_activation_patching import (
        DEFAULT_SEED,
        _capture_all_donors,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
        _score_all_baselines,
        _score_direction,
        _self_patch_check,
    )
    from slam_omni_diagnostics import FORCED_CHOICE_CONDITIONS, load_model, read_rows


LOGGER = logging.getLogger("run_donor_control_patching")
CONTROL_KINDS = ("matched_swap", "same_emotion", "random_donor")
MATCH_KEYS = ("statement", "repetition", "intensity")


def _matched_mapping(rows: list[dict[str, str]], pairs: list[dict[str, Any]]) -> list[int]:
    mapping = list(range(len(rows)))
    for pair in pairs:
        mapping[pair["happy_index"]] = pair["sad_index"]
        mapping[pair["sad_index"]] = pair["happy_index"]
    return mapping


def _same_emotion_mapping(rows: list[dict[str, str]], seed: int) -> list[int]:
    pools: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pools[(row["emotion"], *(row[key] for key in MATCH_KEYS))].append(index)
    rng = random.Random(seed)
    mapping: list[int] = []
    for index, row in enumerate(rows):
        key = (row["emotion"], *(row[key] for key in MATCH_KEYS))
        candidates = [
            candidate
            for candidate in pools[key]
            if candidate != index and rows[candidate]["actor"] != row["actor"]
        ]
        if not candidates:
            raise ValueError(f"No same-emotion cross-actor donor exists for {row['sample_id']}")
        mapping.append(rng.choice(sorted(candidates)))
    return mapping


def _random_mapping(rows: list[dict[str, str]], seed: int) -> list[int]:
    rng = random.Random(seed)
    all_indices = list(range(len(rows)))
    mapping: list[int] = []
    for index in all_indices:
        mapping.append(rng.choice([candidate for candidate in all_indices if candidate != index]))
    return mapping


def build_donor_mapping(rows: list[dict[str, str]], pairs: list[dict[str, Any]], control: str, seed: int) -> list[int]:
    """Build a deterministic target-index to donor-index mapping."""
    if control == "matched_swap":
        return _matched_mapping(rows, pairs)
    if control == "same_emotion":
        return _same_emotion_mapping(rows, seed)
    if control == "random_donor":
        return _random_mapping(rows, seed)
    raise ValueError(f"Unknown donor control: {control}")


def donor_aligned_effect(target_emotion: str, donor_emotion: str, baseline_margin: float, patched_margin: float) -> float:
    """Orient a margin shift toward the donor label."""
    del target_emotion  # The sign is determined by the emotion being donated.
    donor_sign = 1.0 if donor_emotion == "happy" else -1.0
    return donor_sign * (patched_margin - baseline_margin)


def _mapping_rows(
    rows: list[dict[str, str]],
    mapping: list[int],
    control: str,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for target_index, donor_index in enumerate(mapping):
        target, donor = rows[target_index], rows[donor_index]
        output.append(
            {
                "control": control,
                "seed": seed,
                "target_sample_id": target["sample_id"],
                "target_emotion": target["emotion"],
                "target_actor": target["actor"],
                "target_statement": target["statement"],
                "target_repetition": target["repetition"],
                "target_intensity": target["intensity"],
                "donor_sample_id": donor["sample_id"],
                "donor_emotion": donor["emotion"],
                "donor_actor": donor["actor"],
                "donor_statement": donor["statement"],
                "donor_repetition": donor["repetition"],
                "donor_intensity": donor["intensity"],
                "donor_same_emotion": donor["emotion"] == target["emotion"],
                "donor_same_actor": donor["actor"] == target["actor"],
                "donor_same_content": all(donor[key] == target[key] for key in MATCH_KEYS),
                "donor_is_matched_counterpart": donor["pair_id"] == target["pair_id"] and donor["emotion"] != target["emotion"],
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest, args.limit)
    if args.start:
        rows = rows[args.start :]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError("No complete happy/sad pairs were found")
    layers = _llm_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"Invalid layer index {args.layer}; model has {len(layers)} layers")
    condition = FORCED_CHOICE_CONDITIONS[args.condition]
    happy_ids = [int(token_id) for token_id in tokenizer.encode(condition["happy_verbalizer"], add_special_tokens=False)]
    sad_ids = [int(token_id) for token_id in tokenizer.encode(condition["sad_verbalizer"], add_special_tokens=False)]
    if len(happy_ids) != 1 or len(sad_ids) != 1:
        raise ValueError(f"Donor controls require single-token verbalizers: happy={happy_ids}, sad={sad_ids}")
    prefixes, positions = _prepare_prefixes(
        torch, model, whisper, tokenizer, rows, args.ravdess_root, condition["prompt"]
    )
    device = next(model.parameters()).device
    happy_baseline, sad_baseline = _score_all_baselines(
        torch, model, prefixes, positions, happy_ids, sad_ids, device, args.batch_size
    )
    donor_store = _capture_all_donors(
        torch, model, prefixes, positions, (args.layer,), device, args.batch_size
    )
    self_patch_error = _self_patch_check(
        torch, model, prefixes, positions, donor_store, (args.layer,), ("audio_tokens",),
        happy_ids, sad_ids, happy_baseline, sad_baseline, device
    )
    if self_patch_error > args.self_patch_tolerance:
        raise RuntimeError(f"Self-patch changed likelihood by {self_patch_error:.6g}, exceeding tolerance")

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "control", "seed", "pair_id", "actor", "statement", "repetition", "intensity",
        "condition_id", "prompt", "happy_verbalizer", "sad_verbalizer", "happy_token_ids", "sad_token_ids",
        "layer_index", "layer", "patch_site", "happy_sample_id", "sad_sample_id",
        "happy_target_donor_sample_id", "happy_target_donor_emotion", "happy_target_donor_actor",
        "happy_donor_same_emotion", "happy_donor_same_actor", "happy_donor_same_content", "happy_donor_is_matched_counterpart",
        "sad_target_donor_sample_id", "sad_target_donor_emotion", "sad_target_donor_actor",
        "sad_donor_same_emotion", "sad_donor_same_actor", "sad_donor_same_content", "sad_donor_is_matched_counterpart",
        "baseline_happy_margin", "baseline_sad_margin", "patched_happy_target_margin", "patched_sad_target_margin",
        "happy_donor_aligned_effect", "sad_donor_aligned_effect", "counterfactual_effect",
        "happy_direction", "sad_direction", "error",
    ]
    mapping_output = output_path.with_name(output_path.stem + "_mapping.csv")
    with output_path.open("w", newline="") as handle, mapping_output.open("w", newline="") as mapping_handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        mapping_writer = None
        for control in args.control:
            mapping = build_donor_mapping(rows, pairs, control, args.seed)
            map_rows = _mapping_rows(rows, mapping, control, args.seed)
            if mapping_writer is None:
                mapping_writer = csv.DictWriter(mapping_handle, fieldnames=list(map_rows[0]), lineterminator="\n")
                mapping_writer.writeheader()
            mapping_writer.writerows(map_rows)
            donor_indices = mapping
            target_indices = list(range(len(rows)))
            patched_happy, patched_happy_sad = _score_direction(
                torch, model, prefixes, positions, target_indices, donor_indices, donor_store,
                args.layer, "audio_tokens", happy_ids, sad_ids, device, args.batch_size
            )
            # The same target-index score dictionaries contain both candidate
            # likelihoods; build margins for every target before pairing.
            patched_margin = {
                index: patched_happy[index] - patched_happy_sad[index]
                for index in target_indices
            }
            for pair in pairs:
                happy_index, sad_index = pair["happy_index"], pair["sad_index"]
                happy_map, sad_map = map_rows[happy_index], map_rows[sad_index]
                baseline_happy_margin = float(happy_baseline[happy_index] - sad_baseline[happy_index])
                baseline_sad_margin = float(happy_baseline[sad_index] - sad_baseline[sad_index])
                happy_effect = donor_aligned_effect(
                    "happy", happy_map["donor_emotion"], baseline_happy_margin, patched_margin[happy_index]
                )
                sad_effect = donor_aligned_effect(
                    "sad", sad_map["donor_emotion"], baseline_sad_margin, patched_margin[sad_index]
                )
                writer.writerow(
                    {
                        "control": control,
                        "seed": args.seed,
                        "pair_id": pair["pair_id"],
                        "actor": pair["happy"]["actor"],
                        "statement": pair["happy"]["statement"],
                        "repetition": pair["happy"]["repetition"],
                        "intensity": pair["happy"]["intensity"],
                        "condition_id": args.condition,
                        "prompt": condition["prompt"],
                        "happy_verbalizer": condition["happy_verbalizer"],
                        "sad_verbalizer": condition["sad_verbalizer"],
                        "happy_token_ids": json.dumps(happy_ids),
                        "sad_token_ids": json.dumps(sad_ids),
                        "layer_index": args.layer,
                        "layer": f"layer_{args.layer}",
                        "patch_site": "post_decoder_block_output",
                        "happy_sample_id": pair["happy"]["sample_id"],
                        "sad_sample_id": pair["sad"]["sample_id"],
                        "happy_target_donor_sample_id": happy_map["donor_sample_id"],
                        "happy_target_donor_emotion": happy_map["donor_emotion"],
                        "happy_target_donor_actor": happy_map["donor_actor"],
                        "happy_donor_same_emotion": happy_map["donor_same_emotion"],
                        "happy_donor_same_actor": happy_map["donor_same_actor"],
                        "happy_donor_same_content": happy_map["donor_same_content"],
                        "happy_donor_is_matched_counterpart": happy_map["donor_is_matched_counterpart"],
                        "sad_target_donor_sample_id": sad_map["donor_sample_id"],
                        "sad_target_donor_emotion": sad_map["donor_emotion"],
                        "sad_target_donor_actor": sad_map["donor_actor"],
                        "sad_donor_same_emotion": sad_map["donor_same_emotion"],
                        "sad_donor_same_actor": sad_map["donor_same_actor"],
                        "sad_donor_same_content": sad_map["donor_same_content"],
                        "sad_donor_is_matched_counterpart": sad_map["donor_is_matched_counterpart"],
                        "baseline_happy_margin": baseline_happy_margin,
                        "baseline_sad_margin": baseline_sad_margin,
                        "patched_happy_target_margin": patched_margin[happy_index],
                        "patched_sad_target_margin": patched_margin[sad_index],
                        "happy_donor_aligned_effect": happy_effect,
                        "sad_donor_aligned_effect": sad_effect,
                        "counterfactual_effect": 0.5 * (happy_effect + sad_effect),
                        "happy_direction": happy_effect > 0,
                        "sad_direction": sad_effect > 0,
                        "error": "",
                    }
                )
            handle.flush()
            LOGGER.info("finished donor control: %s", control)
    output_path.with_name(output_path.stem + "_run.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "n_samples": len(rows),
                "n_pairs": len(pairs),
                "condition_id": args.condition,
                "happy_token_ids": happy_ids,
                "sad_token_ids": sad_ids,
                "layer_index": args.layer,
                "controls": list(args.control),
                "patch_site": "post_decoder_block_output",
                "audio_start": positions["audio_start"],
                "audio_length": positions["audio_length"],
                "prefix_length": positions["prefix_length"],
                "checkpoint": str(args.checkpoint),
                "model_parameters_frozen": True,
                "self_patch_max_abs_likelihood_error": self_patch_error,
                "self_patch_tolerance": args.self_patch_tolerance,
                "effect_definition": "donor_sign*(patched_target_margin-baseline_target_margin), donor_sign=+1 for happy and -1 for sad",
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    LOGGER.info("saved donor control rows to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--slam-llm-root", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--condition", choices=list(FORCED_CHOICE_CONDITIONS), default="upper_prompt__lower_spaced")
    parser.add_argument("--control", nargs="+", choices=list(CONTROL_KINDS), default=list(CONTROL_KINDS))
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-patch-tolerance", type=float, default=1e-3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.self_patch_tolerance < 0:
        raise SystemExit("--self-patch-tolerance must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
