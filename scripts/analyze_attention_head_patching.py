#!/usr/bin/env python3
"""Summarize targeted causal attention-head patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.analyze_donor_controls import actor_cluster_bootstrap
except ModuleNotFoundError:  # Direct script entry point.
    from analyze_donor_controls import actor_cluster_bootstrap


REQUIRED_COLUMNS = {"pair_id", "actor", "layer_index", "head_index", "counterfactual_effect"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS.difference(rows[0] if rows else REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Attention-head CSV is missing required columns: {sorted(missing)}")
    return rows


def _mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    half_width = 1.96 * sd / math.sqrt(len(values))
    return mean, sd, mean - half_width, mean + half_width


def summarize_head_rows(
    rows: list[dict[str, str]], *, bootstrap_seed: int = 20260915, bootstrap_samples: int = 5000
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Cannot summarize failed head row {row['pair_id']}: {row['error']}")
        grouped[(int(row["layer_index"]), int(row["head_index"]))].append(row)
    summaries = []
    for (layer_index, head_index), group in sorted(grouped.items()):
        values = [float(row["counterfactual_effect"]) for row in group]
        mean, sd, ci_low, ci_high = _mean_ci95(values)
        boot_low, boot_high = actor_cluster_bootstrap(
            values, [row["actor"] for row in group], seed=bootstrap_seed, samples=bootstrap_samples
        )
        summaries.append(
            {
                "layer_index": layer_index,
                "head_index": head_index,
                "n_pairs": len(values),
                "n_actor_clusters": len({row["actor"] for row in group}),
                "mean_counterfactual_effect": mean,
                "sd_counterfactual_effect": sd,
                "ci95_low_normal": ci_low,
                "ci95_high_normal": ci_high,
                "ci95_low_actor_bootstrap": boot_low,
                "ci95_high_actor_bootstrap": boot_high,
                "median_counterfactual_effect": median(values),
                "positive_ce_fraction": sum(value > 0 for value in values) / len(values),
                "happy_direction_fraction": sum(float(row.get("happy_effect_toward_sad", 0.0)) > 0 for row in group) / len(group),
                "sad_direction_fraction": sum(float(row.get("sad_effect_toward_happy", 0.0)) > 0 for row in group) / len(group),
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
    labels = [f"L{row['layer_index']}H{row['head_index']}" for row in summaries]
    means = [float(row["mean_counterfactual_effect"]) for row in summaries]
    low = [float(row["ci95_low_actor_bootstrap"]) for row in summaries]
    high = [float(row["ci95_high_actor_bootstrap"]) for row in summaries]
    yerr = [[mean - lo for mean, lo in zip(means, low)], [hi - mean for mean, hi in zip(means, high)]]
    fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5), constrained_layout=True)
    axis.errorbar(labels, means, yerr=yerr, fmt="o", capsize=3)
    axis.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axis.set_ylabel("mean donor-aligned CE")
    axis.set_title("Targeted causal attention-head patches")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=45)
    fig.savefig(output_dir / "attention_head_patch_summary.png", dpi=180)
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
    summaries = summarize_head_rows(
        rows, bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples
    )
    _write_csv(args.output_dir / "attention_head_patch_summary.csv", summaries)
    (args.output_dir / "attention_head_patch_summary.json").write_text(
        json.dumps(
            {
                "source": str(args.input_csv),
                "n_raw_rows": len(rows),
                "bootstrap": {"unit": "actor cluster", "seed": args.bootstrap_seed, "samples": args.bootstrap_samples},
                "effect_definition": "symmetric donor-aligned forced-choice margin CE",
                "causal_status": "targeted pre-o_proj decision-head state patch",
                "summary_rows": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    _write_plot(args.output_dir, summaries)


if __name__ == "__main__":
    main()
