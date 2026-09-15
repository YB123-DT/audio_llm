#!/usr/bin/env python3
"""Summarize audio-state restoration and decision-state persistence patches."""

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
    "experiment_type",
    "schedule_id",
    "patch_kind",
    "baseline_happy_margin",
    "baseline_sad_margin",
    "patched_happy_target_margin",
    "patched_sad_target_margin",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS.difference(rows[0] if rows else REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Persistence patch CSV is missing required columns: {sorted(missing)}")
    return rows


def _effect_values(row: dict[str, str]) -> tuple[float, float, float]:
    baseline_happy = float(row["baseline_happy_margin"])
    baseline_sad = float(row["baseline_sad_margin"])
    patched_happy = float(row["patched_happy_target_margin"])
    patched_sad = float(row["patched_sad_target_margin"])
    happy_effect = baseline_happy - patched_happy
    sad_effect = patched_sad - baseline_sad
    return happy_effect, sad_effect, 0.5 * (happy_effect + sad_effect)


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    return mean, standard_deviation, 1.96 * standard_deviation / math.sqrt(len(values))


def _schedule_order(row: dict[str, str]) -> float:
    if row["experiment_type"] == "audio_restore":
        value = row.get("restore_layer", "")
        return 24.0 if value in {"", "None", "none"} else float(value)
    return float(row.get("clamp_end_layer", "nan"))


def summarize_persistence_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate the 96 pair-level effects for each persistence schedule."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Cannot summarize failed patch row {row['pair_id']}: {row['error']}")
        grouped[(row["experiment_type"], row["schedule_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (experiment_type, schedule_id), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], _schedule_order(item[1][0]))
    ):
        effects = [_effect_values(row) for row in group]
        happy_effects = [effect[0] for effect in effects]
        sad_effects = [effect[1] for effect in effects]
        ce_values = [effect[2] for effect in effects]
        direction_both = [happy > 0 and sad > 0 for happy, sad in zip(happy_effects, sad_effects)]
        mean_ce, sd_ce, ci_half_width = _mean_ci95(ce_values)
        first = group[0]
        restore_layer = first.get("restore_layer", "")
        clamp_end_layer = first.get("clamp_end_layer", "")
        summaries.append(
            {
                "experiment_type": experiment_type,
                "schedule_id": schedule_id,
                "patch_kind": first["patch_kind"],
                "schedule_order": _schedule_order(first),
                "restore_layer": restore_layer,
                "clamp_end_layer": clamp_end_layer,
                "n_pairs": len(group),
                "mean_counterfactual_effect": mean_ce,
                "sd_counterfactual_effect": sd_ce,
                "ci95_low_counterfactual_effect": mean_ce - ci_half_width,
                "ci95_high_counterfactual_effect": mean_ce + ci_half_width,
                "median_counterfactual_effect": median(ce_values),
                "positive_ce_fraction": sum(value > 0 for value in ce_values) / len(ce_values),
                "mean_happy_effect_toward_sad": sum(happy_effects) / len(happy_effects),
                "mean_sad_effect_toward_happy": sum(sad_effects) / len(sad_effects),
                "happy_direction_fraction": sum(value > 0 for value in happy_effects) / len(happy_effects),
                "sad_direction_fraction": sum(value > 0 for value in sad_effects) / len(sad_effects),
                "both_directions_fraction": sum(direction_both) / len(direction_both),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (output_dir / "persistence_plot_unavailable.txt").write_text(
            "matplotlib is not installed; CSV and JSON outputs remain available.\n"
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for experiment_type, label, axis_index in (
        ("audio_restore", "audio patch: restore target state", 0),
        ("decision_clamp", "decision patch: clamp donor state", 1),
    ):
        group = sorted(
            (row for row in summaries if row["experiment_type"] == experiment_type),
            key=lambda row: float(row["schedule_order"]),
        )
        if not group:
            axes[axis_index].axis("off")
            continue
        x = [float(row["schedule_order"]) for row in group]
        y = [float(row["mean_counterfactual_effect"]) for row in group]
        low = [float(row["ci95_low_counterfactual_effect"]) for row in group]
        high = [float(row["ci95_high_counterfactual_effect"]) for row in group]
        yerr = [[value - lo for value, lo in zip(y, low)], [hi - value for value, hi in zip(y, high)]]
        axes[axis_index].errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.2)
        axes[axis_index].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        axes[axis_index].set_xlabel(
            "restore layer (24 = no restore)" if experiment_type == "audio_restore" else "clamp end layer"
        )
        axes[axis_index].set_ylabel("mean CE in log-likelihood margin")
        axes[axis_index].set_title(label)
        axes[axis_index].grid(alpha=0.25)
        axes[axis_index].set_xticks(x)
        axes[axis_index].set_xticklabels(
            ["no restore" if experiment_type == "audio_restore" and value == 24.0 else str(int(value)) for value in x],
            rotation=35,
            ha="right",
        )
    fig.savefig(output_dir / "persistence_patch_summary.png", dpi=180)
    plt.close(fig)


def _interpretation(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    audio = sorted(
        (row for row in summaries if row["experiment_type"] == "audio_restore"),
        key=lambda row: float(row["schedule_order"]),
    )
    decision = sorted(
        (row for row in summaries if row["experiment_type"] == "decision_clamp"),
        key=lambda row: float(row["schedule_order"]),
    )
    audio_positive_ci = [row for row in audio if float(row["ci95_low_counterfactual_effect"]) > 0]
    decision_positive_ci = [row for row in decision if float(row["ci95_low_counterfactual_effect"]) > 0]
    return {
        "audio_first_descriptive_positive_ci_schedule": audio_positive_ci[0]["schedule_id"] if audio_positive_ci else None,
        "decision_first_descriptive_positive_ci_schedule": decision_positive_ci[0]["schedule_id"] if decision_positive_ci else None,
        "audio_max_mean_schedule": max(audio, key=lambda row: float(row["mean_counterfactual_effect"]))["schedule_id"] if audio else None,
        "decision_max_mean_schedule": max(decision, key=lambda row: float(row["mean_counterfactual_effect"]))["schedule_id"] if decision else None,
        "audio_single_restore_and_no_restore_are_comparable": bool(audio),
        "decision_single_patch_and_persistent_clamp_are_comparable": bool(decision),
        "claim_scope": "Descriptive counterfactual effects under one prompt/verbalizer and one fixed seed; persistence curves do not by themselves prove natural routing.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input_csv)
    summaries = summarize_persistence_rows(rows)
    if not summaries:
        raise ValueError("No successful persistence-patch rows were found")
    interpretation = _interpretation(summaries)
    metadata = {
        "source": str(args.input_csv),
        "n_raw_rows": len(rows),
        "n_summary_rows": len(summaries),
        "schedule_ids": [row["schedule_id"] for row in summaries],
        "n_pairs_per_schedule": sorted({row["n_pairs"] for row in summaries}),
        "condition_ids": sorted({row.get("condition_id", "") for row in rows if row.get("condition_id")}),
        "verbalizers": sorted(
            {
                (row.get("happy_verbalizer", ""), row.get("sad_verbalizer", ""))
                for row in rows
                if row.get("happy_verbalizer") or row.get("sad_verbalizer")
            }
        ),
        "patch_site": "post_decoder_block_output",
        "audio_semantics": "donor audio is patched once at layer 17; target baseline audio is restored at the named later layer, or no restore allows natural evolution",
        "decision_semantics": "donor decision state is clamped at same-layer post-block outputs from layer 17 through the named end layer",
        "effect_definition": "CE=0.5*((baseline_happy_margin-patched_happy_target_margin)+(patched_sad_target_margin-baseline_sad_margin))",
        "ci95_definition": "normal approximation over pair-level effects; descriptive only",
        "multiple_comparison_correction": "none",
        "interpretation": interpretation,
    }
    _write_csv(args.output_dir / "persistence_patch_summary.csv", summaries)
    (args.output_dir / "persistence_patch_summary.json").write_text(
        json.dumps({**metadata, "summary_rows": summaries}, indent=2, ensure_ascii=False) + "\n"
    )
    _write_plot(args.output_dir, summaries)


if __name__ == "__main__":
    main()
