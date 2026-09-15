#!/usr/bin/env python3
"""Summarize emotion-swap and donor-control activation patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


REQUIRED_COLUMNS = {
    "control",
    "pair_id",
    "actor",
    "counterfactual_effect",
    "happy_donor_aligned_effect",
    "sad_donor_aligned_effect",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS.difference(rows[0] if rows else REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Donor-control CSV is missing required columns: {sorted(missing)}")
    return rows


def _mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    half_width = 1.96 * sd / math.sqrt(len(values))
    return mean, sd, mean - half_width, mean + half_width


def actor_cluster_bootstrap(
    values: list[float], actors: list[str], *, seed: int, samples: int = 5000
) -> tuple[float, float]:
    """Return percentile CI after resampling actor clusters with replacement."""
    if len(values) != len(actors):
        raise ValueError("Values and actors must have equal length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, actor in zip(values, actors):
        grouped[actor].append(value)
    actor_ids = sorted(grouped)
    if not actor_ids:
        raise ValueError("No actor clusters were provided")
    actor_means = {actor: sum(grouped[actor]) / len(grouped[actor]) for actor in actor_ids}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rng.choice(actor_ids) for _ in actor_ids]
        estimates.append(sum(actor_means[actor] for actor in draw) / len(draw))
    estimates.sort()
    low_index = max(0, min(len(estimates) - 1, int(0.025 * (len(estimates) - 1))))
    high_index = max(0, min(len(estimates) - 1, int(0.975 * (len(estimates) - 1))))
    return estimates[low_index], estimates[high_index]


def _fraction(rows: list[dict[str, str]], field: str) -> float | None:
    values = [row[field].strip().lower() for row in rows if row.get(field, "") != ""]
    if not values:
        return None
    return sum(value in {"true", "1", "yes"} for value in values) / len(values)


def summarize_control_rows(
    rows: list[dict[str, str]], *, bootstrap_seed: int = 20260915, bootstrap_samples: int = 5000
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Cannot summarize failed donor-control row {row['pair_id']}: {row['error']}")
        grouped[row["control"]].append(row)
    summaries: list[dict[str, Any]] = []
    for control, group in sorted(grouped.items()):
        values = [float(row["counterfactual_effect"]) for row in group]
        actors = [row["actor"] for row in group]
        mean, sd, ci_low, ci_high = _mean_ci95(values)
        bootstrap_low, bootstrap_high = actor_cluster_bootstrap(
            values, actors, seed=bootstrap_seed, samples=bootstrap_samples
        )
        summaries.append(
            {
                "control": control,
                "n_pairs": len(values),
                "n_actor_clusters": len(set(actors)),
                "mean_counterfactual_effect": mean,
                "sd_counterfactual_effect": sd,
                "ci95_low_normal": ci_low,
                "ci95_high_normal": ci_high,
                "ci95_low_actor_bootstrap": bootstrap_low,
                "ci95_high_actor_bootstrap": bootstrap_high,
                "median_counterfactual_effect": median(values),
                "positive_ce_fraction": sum(value > 0 for value in values) / len(values),
                "mean_happy_donor_aligned_effect": sum(float(row["happy_donor_aligned_effect"]) for row in group) / len(group),
                "mean_sad_donor_aligned_effect": sum(float(row["sad_donor_aligned_effect"]) for row in group) / len(group),
                "happy_direction_fraction": sum(float(row["happy_donor_aligned_effect"]) > 0 for row in group) / len(group),
                "sad_direction_fraction": sum(float(row["sad_donor_aligned_effect"]) > 0 for row in group) / len(group),
                "happy_donor_same_emotion_fraction": _fraction(group, "happy_donor_same_emotion"),
                "sad_donor_same_emotion_fraction": _fraction(group, "sad_donor_same_emotion"),
                "happy_donor_same_content_fraction": _fraction(group, "happy_donor_same_content"),
                "sad_donor_same_content_fraction": _fraction(group, "sad_donor_same_content"),
                "happy_donor_is_matched_counterpart_fraction": _fraction(group, "happy_donor_is_matched_counterpart"),
                "sad_donor_is_matched_counterpart_fraction": _fraction(group, "sad_donor_is_matched_counterpart"),
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
        return
    labels = [row["control"] for row in summaries]
    means = [float(row["mean_counterfactual_effect"]) for row in summaries]
    low = [float(row["ci95_low_actor_bootstrap"]) for row in summaries]
    high = [float(row["ci95_high_actor_bootstrap"]) for row in summaries]
    yerr = [[mean - lo for mean, lo in zip(means, low)], [hi - mean for mean, hi in zip(means, high)]]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.errorbar(labels, means, yerr=yerr, fmt="o", capsize=4)
    axis.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axis.set_ylabel("mean donor-aligned CE")
    axis.set_title("Layer17 audio donor controls (actor-cluster bootstrap CI)")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "donor_control_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input_csv)
    summaries = summarize_control_rows(
        rows, bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples
    )
    _write_csv(args.output_dir / "donor_control_summary.csv", summaries)
    metadata = {
        "source": str(args.input_csv),
        "n_raw_rows": len(rows),
        "controls": [row["control"] for row in summaries],
        "bootstrap": {
            "unit": "actor cluster",
            "seed": args.bootstrap_seed,
            "samples": args.bootstrap_samples,
            "percentile_interval": [0.025, 0.975],
        },
        "effect_definition": "donor_sign*(patched_target_margin-baseline_target_margin), donor_sign=+1 for happy and -1 for sad",
        "interpretation": "Matched swap is the emotion-change reference; same-emotion and random donors test broad state replacement controls.",
    }
    (args.output_dir / "donor_control_summary.json").write_text(
        json.dumps({**metadata, "summary_rows": summaries}, indent=2, ensure_ascii=False) + "\n"
    )
    _write_plot(args.output_dir, summaries)


if __name__ == "__main__":
    main()
