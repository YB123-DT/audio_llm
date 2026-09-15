#!/usr/bin/env python3
"""Summarize candidate-token activation-patching effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


REQUIRED_COLUMNS = {
    "pair_id",
    "layer_index",
    "layer",
    "patch_kind",
    "token_position",
    "counterfactual_effect",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS.difference(rows[0] if rows else REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Tokenwise CSV is missing required columns: {sorted(missing)}")
    return rows


def _read_sequence_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"pair_id", "layer_index", "patch_kind", "counterfactual_effect"}
    missing = required.difference(rows[0] if rows else required)
    if missing:
        raise ValueError(f"Sequence-level CSV is missing required columns: {sorted(missing)}")
    return rows


def _mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, mean - half_width, mean + half_width


def summarize_tokenwise_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate CE by layer, patch site and candidate token position."""
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Cannot summarize failed tokenwise row {row['pair_id']}: {row['error']}")
        grouped[(int(row["layer_index"]), row["patch_kind"], int(row["token_position"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (layer_index, patch_kind, token_position), group in sorted(grouped.items()):
        values = [float(row["counterfactual_effect"]) for row in group]
        mean, sd, ci_low, ci_high = _mean_ci95(values)
        summaries.append(
            {
                "layer_index": layer_index,
                "layer": group[0]["layer"],
                "patch_kind": patch_kind,
                "token_position": token_position,
                "n_pairs": len({row["pair_id"] for row in group}),
                "mean_counterfactual_effect": mean,
                "sd_counterfactual_effect": sd,
                "ci95_low_counterfactual_effect": ci_low,
                "ci95_high_counterfactual_effect": ci_high,
                "median_counterfactual_effect": median(values),
                "positive_ce_fraction": sum(value > 0 for value in values) / len(values),
            }
        )
    return summaries


def reconstruct_sequence_effects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Sum token effects per pair to reconstruct sequence-level CE."""
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["layer_index"]), row["patch_kind"], row["pair_id"])
        grouped[key].append(float(row["counterfactual_effect"]))
        metadata.setdefault(key, row)
    by_condition: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (layer_index, patch_kind, _pair_id), values in grouped.items():
        by_condition[(layer_index, patch_kind)].append(sum(values))
    output: list[dict[str, Any]] = []
    for (layer_index, patch_kind), values in sorted(by_condition.items()):
        mean, sd, ci_low, ci_high = _mean_ci95(values)
        output.append(
            {
                "layer_index": layer_index,
                "layer": next(
                    metadata[(layer_index, patch_kind, pair_id)]["layer"]
                    for pair_id in sorted({key[2] for key in grouped if key[:2] == (layer_index, patch_kind)})
                ),
                "patch_kind": patch_kind,
                "n_pairs": len(values),
                "mean_sequence_counterfactual_effect": mean,
                "sd_sequence_counterfactual_effect": sd,
                "ci95_low_sequence_counterfactual_effect": ci_low,
                "ci95_high_sequence_counterfactual_effect": ci_high,
            }
        )
    return output


def compare_sequence_reconstruction(
    rows: list[dict[str, str]], sequence_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Compare summed token CE with an existing sequence-level CSV."""
    token_sums: dict[tuple[int, str, str], float] = defaultdict(float)
    for row in rows:
        token_sums[(int(row["layer_index"]), row["patch_kind"], row["pair_id"])] += float(row["counterfactual_effect"])
    sequence_values: dict[tuple[int, str, str], float] = {}
    for row in sequence_rows:
        sequence_values[(int(row["layer_index"]), row["patch_kind"], row["pair_id"])] = float(row["counterfactual_effect"])
    conditions = sorted({key[:2] for key in token_sums})
    output: list[dict[str, Any]] = []
    for layer_index, patch_kind in conditions:
        keys = [key for key in token_sums if key[:2] == (layer_index, patch_kind)]
        differences = [token_sums[key] - sequence_values[key] for key in keys if key in sequence_values]
        if not differences:
            continue
        output.append(
            {
                "layer_index": layer_index,
                "patch_kind": patch_kind,
                "n_pairs_compared": len(differences),
                "max_abs_reconstruction_error": max(abs(value) for value in differences),
                "mean_abs_reconstruction_error": sum(abs(value) for value in differences) / len(differences),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-csv", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input_csv)
    summaries = summarize_tokenwise_rows(rows)
    reconstruction = reconstruct_sequence_effects(rows)
    comparison = []
    if args.sequence_csv:
        comparison = compare_sequence_reconstruction(rows, _read_sequence_rows(args.sequence_csv))
    _write_csv(args.output_dir / "tokenwise_activation_patch_summary.csv", summaries)
    _write_csv(args.output_dir / "tokenwise_activation_patch_reconstruction.csv", reconstruction)
    _write_csv(args.output_dir / "tokenwise_activation_patch_sequence_comparison.csv", comparison)
    metadata = {
        "source": str(args.input_csv),
        "n_raw_rows": len(rows),
        "n_summary_rows": len(summaries),
        "token_positions": sorted({int(row["token_position"]) for row in rows}),
        "sequence_csv": str(args.sequence_csv) if args.sequence_csv else None,
        "effect_definition": "per-token CE=0.5*((baseline_happy_margin-patched_happy_target_margin)+(patched_sad_target_margin-baseline_sad_margin)); summing positions reconstructs sequence CE",
        "ci95_definition": "normal approximation over pair-level effects within each token position",
    }
    (args.output_dir / "tokenwise_activation_patch_summary.json").write_text(
        json.dumps({**metadata, "token_summaries": summaries, "sequence_reconstruction": reconstruction, "sequence_comparison": comparison}, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
