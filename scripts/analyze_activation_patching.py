#!/usr/bin/env python3
"""Summarize matched-pair activation-patching effects."""

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
        raise ValueError(f"Activation patch CSV is missing required columns: {sorted(missing)}")
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
    """Return mean, sample SD, and a normal-approximation 95% CI."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, half_width


def summarize_patch_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate bidirectional donor effects by layer and patch kind."""
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Cannot summarize failed patch row {row['pair_id']}: {row['error']}")
        grouped[(int(row["layer_index"]), row["patch_kind"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (layer_index, patch_kind), group in sorted(grouped.items()):
        effects = [_effect_values(row) for row in group]
        happy_effects = [effect[0] for effect in effects]
        sad_effects = [effect[1] for effect in effects]
        ce_values = [effect[2] for effect in effects]
        direction_both = [
            happy_effect > 0 and sad_effect > 0
            for happy_effect, sad_effect in zip(happy_effects, sad_effects)
        ]
        mean_ce, sd_ce, ci95_half_width = _mean_ci95(ce_values)
        summaries.append(
            {
                "layer_index": layer_index,
                "layer": group[0]["layer"],
                "patch_kind": patch_kind,
                "n_pairs": len(group),
                "mean_counterfactual_effect": mean_ce,
                "sd_counterfactual_effect": sd_ce,
                "ci95_low_counterfactual_effect": mean_ce - ci95_half_width,
                "ci95_high_counterfactual_effect": mean_ce + ci95_half_width,
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


def mechanism_label(audio_effective: bool, decision_effective: bool) -> str:
    if audio_effective and decision_effective:
        return "audio 与 decision patch 的 CE 均为正且 CI95 下界 > 0；与 downstream readout 可用、自然 routing/权重不足相一致"
    if not audio_effective and decision_effective:
        return "decision patch 的 CE CI95 下界 > 0 而 audio patch 未达到；与 audio→decision routing failure 相一致"
    if not audio_effective and not decision_effective:
        return "两类 patch 的 CE 都未达到 CI95 下界 > 0；不能据此区分 downstream readout、标签映射或 patch state 偏离分布"
    return "audio patch 的 CE CI95 下界 > 0 而 decision patch 未达到；提示作用可能依赖分布式 token computation，或 decision patch 产生 off-manifold state"


def build_mechanism_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for summary in summaries:
        by_layer[int(summary["layer_index"])][summary["patch_kind"]] = summary
    rows: list[dict[str, Any]] = []
    for layer_index in sorted(by_layer):
        group = by_layer[layer_index]
        if "audio_tokens" not in group or "decision_token" not in group:
            continue
        audio = group["audio_tokens"]
        decision = group["decision_token"]
        audio_effective = float(audio["ci95_low_counterfactual_effect"]) > 0
        decision_effective = float(decision["ci95_low_counterfactual_effect"]) > 0
        rows.append(
            {
                "layer_index": layer_index,
                "layer": audio["layer"],
                "audio_mean_counterfactual_effect": audio["mean_counterfactual_effect"],
                "decision_mean_counterfactual_effect": decision["mean_counterfactual_effect"],
                "audio_positive_ce_fraction": audio["positive_ce_fraction"],
                "decision_positive_ce_fraction": decision["positive_ce_fraction"],
                "audio_both_directions_fraction": audio["both_directions_fraction"],
                "decision_both_directions_fraction": decision["both_directions_fraction"],
                "audio_effective_by_mean_sign": float(audio["mean_counterfactual_effect"]) > 0,
                "decision_effective_by_mean_sign": float(decision["mean_counterfactual_effect"]) > 0,
                "audio_effective_by_ci95": audio_effective,
                "decision_effective_by_ci95": decision_effective,
                "interpretation": mechanism_label(audio_effective, decision_effective),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(output_dir: Path, summaries: list[dict[str, Any]], mechanism_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (output_dir / "plot_unavailable.txt").write_text(
            "matplotlib is not installed; CSV and JSON outputs remain available.\n"
        )
        return

    layers = sorted({int(summary["layer_index"]) for summary in summaries})
    by_kind = {
        kind: {int(summary["layer_index"]): summary for summary in summaries if summary["patch_kind"] == kind}
        for kind in ("audio_tokens", "decision_token")
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for kind, label in (("audio_tokens", "audio-token patch"), ("decision_token", "decision-token patch")):
        available_layers = [layer for layer in layers if layer in by_kind[kind]]
        if not available_layers:
            continue
        values = [by_kind[kind][layer]["mean_counterfactual_effect"] for layer in available_layers]
        axes[0, 0].plot(available_layers, values, marker=".", label=label)
        directions = [by_kind[kind][layer]["both_directions_fraction"] for layer in available_layers]
        axes[0, 1].plot(available_layers, directions, marker=".", label=label)
    axes[0, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Mean counterfactual effect", xlabel="LLM layer", ylabel="CE in log-likelihood margin")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Both directions donor-aligned", xlabel="LLM layer", ylabel="Fraction")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    audio_effects = [row["audio_mean_counterfactual_effect"] for row in mechanism_rows]
    decision_effects = [row["decision_mean_counterfactual_effect"] for row in mechanism_rows]
    mechanism_layers = [row["layer_index"] for row in mechanism_rows]
    axes[1, 0].plot(mechanism_layers, audio_effects, marker=".", label="audio-token patch")
    axes[1, 0].plot(mechanism_layers, decision_effects, marker=".", label="decision-token patch")
    axes[1, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Audio vs decision patch effect", xlabel="LLM layer", ylabel="Mean CE")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "CI-based mechanism labels are descriptive.\n"
        "Positive CE means the target margin moves toward\n"
        "the donor emotion in both swap directions.",
        va="top",
        fontsize=10,
    )
    fig.savefig(output_dir / "activation_patch_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input_csv)
    summaries = summarize_patch_rows(rows)
    mechanisms = build_mechanism_rows(summaries)
    if not summaries:
        raise ValueError("No successful activation-patch rows were found")
    summary = {
        "source": str(args.input_csv),
        "n_raw_rows": len(rows),
        "n_summary_rows": len(summaries),
        "n_layers": len({row["layer_index"] for row in summaries}),
        "n_pairs_per_condition": sorted({row["n_pairs"] for row in summaries}),
        "condition_ids": sorted({row.get("condition_id", "") for row in rows if row.get("condition_id")}),
        "verbalizers": sorted(
            {
                (row.get("happy_verbalizer", ""), row.get("sad_verbalizer", ""))
                for row in rows
                if row.get("happy_verbalizer") or row.get("sad_verbalizer")
            }
        ),
        "patch_site": "post_decoder_block_output",
        "effect_definition": "CE=0.5*((baseline_happy_margin-patched_happy_target_margin)+(patched_sad_target_margin-baseline_sad_margin))",
        "mechanism_rule": "effective iff descriptive normal-approximation CI95 lower bound > 0; mean and direction fractions remain reported",
        "ci95_definition": "normal approximation over the 96 pair-level counterfactual effects; descriptive only",
        "interpretation": "Activation patching measures causal counterfactual influence under the selected layer, position, prompt, and verbalizer; it does not by itself establish natural routing.",
        "layers": sorted({row["layer_index"] for row in summaries}),
    }
    _write_csv(args.output_dir / "activation_patch_summary.csv", summaries)
    _write_csv(args.output_dir / "activation_patch_mechanism.csv", mechanisms)
    (args.output_dir / "activation_patch_summary.json").write_text(
        json.dumps({**summary, "mechanism_rows": mechanisms}, indent=2, ensure_ascii=False) + "\n"
    )
    _write_plot(args.output_dir, summaries, mechanisms)


if __name__ == "__main__":
    main()
