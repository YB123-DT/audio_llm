#!/usr/bin/env python3
"""Measure how long matched-pair emotion interventions remain decision-relevant.

The experiment has two deliberately different schedules:

``audio_restore``
    Patch the target's audio-token state with the matched donor at the
    post-output of ``initial_layer`` (17).  Optionally restore the target's
    *unpatched* audio-token state at a later post-block layer.  The no-restore
    schedule applies only the initial patch and lets subsequent decoder blocks
    evolve naturally.

``decision_clamp``
    Replace the target decision position with the matched donor state at every
    post-block layer from ``initial_layer`` through ``clamp_end_layer``.  An end
    layer of 17 is the single-patch control; later end layers are persistent
    clamps using the donor state captured at the corresponding layer.

All scores use the same teacher-forced HAPPY/SAD sequence likelihood and the
same symmetric counterfactual margin used by ``run_activation_patching.py``.
No model parameters are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.run_activation_patching import (
        DEFAULT_SEED,
        PATCH_KINDS,
        _capture_all_donors,
        _counterfactual_margin_effects,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
        _score_all_baselines,
        _score_candidate_pairs,
    )
    from scripts.slam_omni_diagnostics import (
        FORCED_CHOICE_CONDITIONS,
        load_model,
        read_rows,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_persistence_patching.py`` entry point.
    from run_activation_patching import (
        DEFAULT_SEED,
        PATCH_KINDS,
        _capture_all_donors,
        _counterfactual_margin_effects,
        _llm_layers,
        _pair_indices,
        _prepare_prefixes,
        _score_all_baselines,
        _score_candidate_pairs,
    )
    from slam_omni_diagnostics import (
        FORCED_CHOICE_CONDITIONS,
        load_model,
        read_rows,
    )


LOGGER = logging.getLogger("run_persistence_patching")
INITIAL_LAYER = 17
AUDIO_RESTORE_LAYERS = (18, 19, 20, 21, 22, 23)
DECISION_CLAMP_END_LAYERS = (17, 18, 19, 20, 21, 22, 23)
SCHEDULE_KINDS = ("audio_restore", "decision_clamp")
StateSchedule = tuple[tuple[int, str, str], ...]


def build_audio_restore_schedule(
    restore_layer: int | None,
    *,
    initial_layer: int = INITIAL_LAYER,
) -> StateSchedule:
    """Return ``(layer, kind, source)`` entries for one audio schedule.

    The source is ``donor`` for the initial intervention and ``target`` for a
    later restore.  States are captured from independent, unpatched forwards;
    between hooks the target stream evolves normally through the decoder.
    """
    if restore_layer is not None and restore_layer <= initial_layer:
        raise ValueError("audio restore layer must be after the initial patch layer")
    entries: list[tuple[int, str, str]] = [(initial_layer, "audio_tokens", "donor")]
    if restore_layer is not None:
        entries.append((restore_layer, "audio_tokens", "target"))
    return tuple(entries)


def build_decision_clamp_schedule(
    clamp_end_layer: int,
    *,
    initial_layer: int = INITIAL_LAYER,
) -> StateSchedule:
    """Return same-layer donor decision clamps from the initial layer onward."""
    if clamp_end_layer < initial_layer:
        raise ValueError("decision clamp end layer must be at or after the initial layer")
    return tuple(
        (layer_index, "decision_token", "donor")
        for layer_index in range(initial_layer, clamp_end_layer + 1)
    )


def schedule_id(mode: str, *, restore_layer: int | None, clamp_end_layer: int | None) -> str:
    if mode == "audio_restore":
        return "audio_no_restore" if restore_layer is None else f"audio_restore_at_{restore_layer}"
    if mode == "decision_clamp":
        if clamp_end_layer is None:
            raise ValueError("decision clamp requires --clamp-end-layer")
        return f"decision_clamp_to_{clamp_end_layer}"
    raise ValueError(f"Unknown persistence mode: {mode}")


def _schedule_for_args(args: argparse.Namespace) -> tuple[str, StateSchedule]:
    if args.mode == "audio_restore":
        if args.clamp_end_layer is not None:
            raise ValueError("--clamp-end-layer is only valid for decision_clamp")
        return schedule_id(args.mode, restore_layer=args.restore_layer, clamp_end_layer=None), build_audio_restore_schedule(
            args.restore_layer,
            initial_layer=args.initial_layer,
        )
    if args.restore_layer is not None:
        raise ValueError("--restore-layer is only valid for audio_restore")
    if args.clamp_end_layer is None:
        raise ValueError("decision_clamp requires --clamp-end-layer")
    return schedule_id(args.mode, restore_layer=None, clamp_end_layer=args.clamp_end_layer), build_decision_clamp_schedule(
        args.clamp_end_layer,
        initial_layer=args.initial_layer,
    )


def _score_schedule_direction(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    target_indices: list[int],
    donor_indices: list[int],
    state_store: dict[int, dict[str, Any]],
    schedule: StateSchedule,
    happy_ids: list[int],
    sad_ids: list[int],
    device: Any,
    batch_size: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Score one target←donor direction with all schedule hooks installed."""
    if len(target_indices) != len(donor_indices):
        raise ValueError("Target and donor index lists differ in length")
    layers = _llm_layers(model)
    happy_scores: dict[int, float] = {}
    sad_scores: dict[int, float] = {}
    for start in range(0, len(target_indices), batch_size):
        target_chunk = target_indices[start : start + batch_size]
        donor_chunk = donor_indices[start : start + batch_size]
        prefix_batch = torch.stack([prefixes[index] for index in target_chunk]).to(device)
        patch_specs: list[tuple[Any, Any, str]] = []
        for layer_index, patch_kind, source in schedule:
            source_indices = donor_chunk if source == "donor" else target_chunk
            if source not in {"donor", "target"}:
                raise ValueError(f"Unknown schedule state source: {source}")
            patch_specs.append(
                (
                    layers[layer_index],
                    state_store[layer_index][patch_kind][source_indices].to(device),
                    patch_kind,
                )
            )
        attention_mask = torch.ones(
            (prefix_batch.shape[0], prefix_batch.shape[1]),
            dtype=torch.bool,
            device=device,
        )
        happy, sad = _score_candidate_pairs(
            torch,
            model,
            prefix_batch,
            attention_mask,
            happy_ids,
            sad_ids,
            positions["prefix_length"],
            audio_start=positions["audio_start"],
            audio_length=positions["audio_length"],
            decision_position=positions["decision_position"],
            patch_specs=patch_specs,
        )
        for offset, index in enumerate(target_chunk):
            happy_scores[index] = float(happy[offset])
            sad_scores[index] = float(sad[offset])
        LOGGER.info(
            "patched schedule %s: %d/%d",
            ",".join(f"{layer}:{kind}:{source}" for layer, kind, source in schedule),
            min(start + batch_size, len(target_indices)),
            len(target_indices),
        )
    return happy_scores, sad_scores


def _self_patch_schedule_check(
    torch: Any,
    model: Any,
    prefixes: list[Any],
    positions: dict[str, int],
    state_store: dict[int, dict[str, Any]],
    schedule: StateSchedule,
    happy_ids: list[int],
    sad_ids: list[int],
    baseline_happy: np.ndarray,
    baseline_sad: np.ndarray,
    device: Any,
) -> float:
    """Verify that a schedule using the target's own states is a no-op."""
    layers = _llm_layers(model)
    prefix_batch = torch.stack([prefixes[0]]).to(device)
    patch_specs = [
        (
            layers[layer_index],
            state_store[layer_index][patch_kind][:1].to(device),
            patch_kind,
        )
        for layer_index, patch_kind, _source in schedule
    ]
    attention_mask = torch.ones((1, prefix_batch.shape[1]), dtype=torch.bool, device=device)
    happy, sad = _score_candidate_pairs(
        torch,
        model,
        prefix_batch,
        attention_mask,
        happy_ids,
        sad_ids,
        positions["prefix_length"],
        audio_start=positions["audio_start"],
        audio_length=positions["audio_length"],
        decision_position=positions["decision_position"],
        patch_specs=patch_specs,
    )
    return max(
        abs(float(happy[0]) - float(baseline_happy[0])),
        abs(float(sad[0]) - float(baseline_sad[0])),
    )


def _validate_schedule(schedule: StateSchedule, n_layers: int) -> None:
    seen_layers: set[int] = set()
    for layer_index, patch_kind, source in schedule:
        if layer_index < 0 or layer_index >= n_layers:
            raise ValueError(f"Invalid layer index {layer_index}; model has {n_layers} layers")
        if layer_index in seen_layers:
            raise ValueError(f"Schedule contains duplicate layer {layer_index}")
        seen_layers.add(layer_index)
        if patch_kind not in PATCH_KINDS:
            raise ValueError(f"Unknown patch kind {patch_kind}")
        if source not in {"donor", "target"}:
            raise ValueError(f"Unknown schedule state source {source}")


def run(args: argparse.Namespace) -> None:
    torch, model, tokenizer, whisper = load_model(args)
    rows = read_rows(args.manifest, args.limit)
    if args.start:
        rows = rows[args.start :]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError("No complete happy/sad pairs were found")

    schedule_name, schedule = _schedule_for_args(args)
    layers = _llm_layers(model)
    _validate_schedule(schedule, len(layers))
    layer_indices = tuple(sorted({entry[0] for entry in schedule}))
    condition = FORCED_CHOICE_CONDITIONS[args.condition]
    happy_ids = [int(token_id) for token_id in tokenizer.encode(condition["happy_verbalizer"], add_special_tokens=False)]
    sad_ids = [int(token_id) for token_id in tokenizer.encode(condition["sad_verbalizer"], add_special_tokens=False)]
    if len(happy_ids) != len(sad_ids):
        raise ValueError(f"Selected verbalizers have different token lengths: happy={happy_ids}, sad={sad_ids}")

    prefixes, positions = _prepare_prefixes(
        torch,
        model,
        whisper,
        tokenizer,
        rows,
        args.ravdess_root,
        condition["prompt"],
    )
    device = next(model.parameters()).device
    happy_baseline, sad_baseline = _score_all_baselines(
        torch,
        model,
        prefixes,
        positions,
        happy_ids,
        sad_ids,
        device,
        args.batch_size,
    )
    state_store = _capture_all_donors(
        torch,
        model,
        prefixes,
        positions,
        layer_indices,
        device,
        args.batch_size,
    )
    self_patch_error = _self_patch_schedule_check(
        torch,
        model,
        prefixes,
        positions,
        state_store,
        schedule,
        happy_ids,
        sad_ids,
        happy_baseline,
        sad_baseline,
        device,
    )
    LOGGER.info("self-patch max absolute likelihood error: %.6g", self_patch_error)
    if self_patch_error > args.self_patch_tolerance:
        raise RuntimeError(
            f"Self-patch control changed likelihood by {self_patch_error:.6g}, "
            f"exceeding tolerance {args.self_patch_tolerance:.6g}"
        )

    sad_targets = [pair["sad_index"] for pair in pairs]
    happy_donors = [pair["happy_index"] for pair in pairs]
    happy_targets = [pair["happy_index"] for pair in pairs]
    sad_donors = [pair["sad_index"] for pair in pairs]
    patched_sad_happy_scores, patched_sad_sad_scores = _score_schedule_direction(
        torch,
        model,
        prefixes,
        positions,
        sad_targets,
        happy_donors,
        state_store,
        schedule,
        happy_ids,
        sad_ids,
        device,
        args.batch_size,
    )
    patched_happy_happy_scores, patched_happy_sad_scores = _score_schedule_direction(
        torch,
        model,
        prefixes,
        positions,
        happy_targets,
        sad_donors,
        state_store,
        schedule,
        happy_ids,
        sad_ids,
        device,
        args.batch_size,
    )

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "actor",
        "statement",
        "repetition",
        "intensity",
        "happy_sample_id",
        "sad_sample_id",
        "condition_id",
        "prompt",
        "happy_verbalizer",
        "sad_verbalizer",
        "happy_token_ids",
        "sad_token_ids",
        "experiment_type",
        "schedule_id",
        "initial_patch_layer",
        "restore_layer",
        "clamp_end_layer",
        "schedule_entries",
        "patch_site",
        "patch_kind",
        "baseline_happy_happy_score",
        "baseline_happy_sad_score",
        "baseline_sad_happy_score",
        "baseline_sad_sad_score",
        "baseline_happy_margin",
        "baseline_sad_margin",
        "patched_happy_target_happy_score",
        "patched_happy_target_sad_score",
        "patched_sad_target_happy_score",
        "patched_sad_target_sad_score",
        "patched_happy_target_margin",
        "patched_sad_target_margin",
        "happy_effect_toward_sad",
        "sad_effect_toward_happy",
        "counterfactual_effect",
        "happy_direction",
        "sad_direction",
        "error",
    ]
    entries_json = json.dumps(
        [
            {"layer": layer_index, "patch_kind": patch_kind, "source": source}
            for layer_index, patch_kind, source in schedule
        ],
        separators=(",", ":"),
    )
    restore_layer = args.restore_layer if args.mode == "audio_restore" else ""
    clamp_end_layer = args.clamp_end_layer if args.mode == "decision_clamp" else ""
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            happy_index = pair["happy_index"]
            sad_index = pair["sad_index"]
            effects = _counterfactual_margin_effects(
                baseline_happy_happy_score=float(happy_baseline[happy_index]),
                baseline_happy_sad_score=float(sad_baseline[happy_index]),
                baseline_sad_happy_score=float(happy_baseline[sad_index]),
                baseline_sad_sad_score=float(sad_baseline[sad_index]),
                patched_happy_target_happy_score=patched_happy_happy_scores[happy_index],
                patched_happy_target_sad_score=patched_happy_sad_scores[happy_index],
                patched_sad_target_happy_score=patched_sad_happy_scores[sad_index],
                patched_sad_target_sad_score=patched_sad_sad_scores[sad_index],
            )
            writer.writerow(
                {
                    "pair_id": pair["pair_id"],
                    "actor": pair["happy"]["actor"],
                    "statement": pair["happy"]["statement"],
                    "repetition": pair["happy"]["repetition"],
                    "intensity": pair["happy"]["intensity"],
                    "happy_sample_id": pair["happy"]["sample_id"],
                    "sad_sample_id": pair["sad"]["sample_id"],
                    "condition_id": args.condition,
                    "prompt": condition["prompt"],
                    "happy_verbalizer": condition["happy_verbalizer"],
                    "sad_verbalizer": condition["sad_verbalizer"],
                    "happy_token_ids": json.dumps(happy_ids),
                    "sad_token_ids": json.dumps(sad_ids),
                    "experiment_type": args.mode,
                    "schedule_id": schedule_name,
                    "initial_patch_layer": args.initial_layer,
                    "restore_layer": restore_layer,
                    "clamp_end_layer": clamp_end_layer,
                    "schedule_entries": entries_json,
                    "patch_site": "post_decoder_block_output",
                    "patch_kind": "audio_tokens" if args.mode == "audio_restore" else "decision_token",
                    "baseline_happy_happy_score": float(happy_baseline[happy_index]),
                    "baseline_happy_sad_score": float(sad_baseline[happy_index]),
                    "baseline_sad_happy_score": float(happy_baseline[sad_index]),
                    "baseline_sad_sad_score": float(sad_baseline[sad_index]),
                    "baseline_happy_margin": effects["baseline_happy_margin"],
                    "baseline_sad_margin": effects["baseline_sad_margin"],
                    "patched_happy_target_happy_score": patched_happy_happy_scores[happy_index],
                    "patched_happy_target_sad_score": patched_happy_sad_scores[happy_index],
                    "patched_sad_target_happy_score": patched_sad_happy_scores[sad_index],
                    "patched_sad_target_sad_score": patched_sad_sad_scores[sad_index],
                    "patched_happy_target_margin": effects["patched_happy_target_margin"],
                    "patched_sad_target_margin": effects["patched_sad_target_margin"],
                    "happy_effect_toward_sad": effects["happy_effect_toward_sad"],
                    "sad_effect_toward_happy": effects["sad_effect_toward_happy"],
                    "counterfactual_effect": effects["counterfactual_effect"],
                    "happy_direction": effects["happy_effect_toward_sad"] > 0,
                    "sad_direction": effects["sad_effect_toward_happy"] > 0,
                    "error": "",
                }
            )
        handle.flush()

    run_metadata = {
        "manifest": str(args.manifest),
        "n_samples": len(rows),
        "n_pairs": len(pairs),
        "condition_id": args.condition,
        "happy_token_ids": happy_ids,
        "sad_token_ids": sad_ids,
        "experiment_type": args.mode,
        "schedule_id": schedule_name,
        "schedule": [
            {"layer": layer_index, "patch_kind": patch_kind, "source": source}
            for layer_index, patch_kind, source in schedule
        ],
        "initial_patch_layer": args.initial_layer,
        "restore_layer": args.restore_layer,
        "clamp_end_layer": args.clamp_end_layer,
        "patch_site": "post_decoder_block_output",
        "audio_start": positions["audio_start"],
        "audio_length": positions["audio_length"],
        "prefix_length": positions["prefix_length"],
        "decision_position": positions["decision_position"],
        "checkpoint": str(args.checkpoint),
        "model_parameters_frozen": True,
        "self_patch_max_abs_likelihood_error": self_patch_error,
        "self_patch_tolerance": args.self_patch_tolerance,
        "seed": args.seed,
        "audio_semantics": "initial donor patch at layer 17; optional target baseline restore at the named later layer; no-restore lets later blocks evolve naturally",
        "decision_semantics": "same-layer donor decision states are clamped at every layer from 17 through clamp_end_layer",
    }
    output_path.with_name(output_path.stem + "_run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n"
    )
    LOGGER.info("saved persistence patch rows to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--slam-llm-root", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--whisper-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--condition", choices=list(FORCED_CHOICE_CONDITIONS), default="upper_prompt__upper_spaced")
    parser.add_argument("--mode", choices=list(SCHEDULE_KINDS), required=True)
    parser.add_argument("--initial-layer", type=int, default=INITIAL_LAYER)
    parser.add_argument("--restore-layer", type=int)
    parser.add_argument("--clamp-end-layer", type=int)
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
    if args.initial_layer < 0:
        raise SystemExit("--initial-layer must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
