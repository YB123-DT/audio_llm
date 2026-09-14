#!/usr/bin/env python3
"""Locate layerwise emotion-probe loss between audio and decision states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROBE_COLUMNS = ("speaker_held_out_accuracy", "statement_held_out_mean")


def _read_probe_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"family", "layer_index", "layer", *PROBE_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Probe CSV is missing required columns: {sorted(required)}")
    return rows


def _retention(decision: float, audio: float) -> float | None:
    above_chance = audio - 0.5
    return (decision - 0.5) / above_chance if above_chance > 0 else None


def build_transfer_rows(probe_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    audio = {int(row["layer_index"]): row for row in probe_rows if row["family"] == "llm_audio_mean"}
    decision = {int(row["layer_index"]): row for row in probe_rows if row["family"] == "llm_decision"}
    if not audio or set(audio) != set(decision):
        raise ValueError("llm_audio_mean and llm_decision layers are not aligned")
    rows: list[dict[str, Any]] = []
    for layer_index in sorted(audio):
        audio_row = audio[layer_index]
        decision_row = decision[layer_index]
        row: dict[str, Any] = {
            "layer_index": layer_index,
            "layer": audio_row["layer"],
        }
        for split, column in (("speaker", "speaker_held_out_accuracy"), ("statement", "statement_held_out_mean")):
            audio_value = float(audio_row[column])
            decision_value = float(decision_row[column])
            previous_decision = None
            if layer_index > 0:
                previous_decision = float(decision[layer_index - 1][column])
            row[f"audio_{split}_probe"] = audio_value
            row[f"decision_{split}_probe"] = decision_value
            row[f"{split}_gap_audio_minus_decision"] = audio_value - decision_value
            row[f"{split}_decision_step"] = decision_value - previous_decision if previous_decision is not None else None
            row[f"{split}_above_chance_retention"] = _retention(decision_value, audio_value)
        rows.append(row)
    return rows


def _largest(rows: list[dict[str, Any]], key: str, predicate: Any = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if row[key] is not None and (predicate(row) if predicate else True)]
    return max(candidates, key=lambda row: row[key]) if candidates else None


def _most_negative(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row[key] is not None]
    return min(candidates, key=lambda row: row[key]) if candidates else None


def summarize_transfer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate losses after the embedding baseline.

    The embedding row is a useful baseline, but it is not a transition between
    LLM layers and therefore should not be reported as an audio-to-decision
    bottleneck.
    """
    model_rows = [row for row in rows if row["layer_index"] > 0]
    return {
        "n_layers": len(rows),
        "chance_baseline": 0.5,
        "embedding_baseline": rows[0] if rows and rows[0]["layer_index"] == 0 else None,
        "bottleneck_search_excludes_layer_indices": [row["layer_index"] for row in rows if row not in model_rows],
        "largest_negative_decision_step": {
            split: _most_negative(model_rows, f"{split}_decision_step") for split in ("speaker", "statement")
        },
        "largest_audio_minus_decision_gap": {
            split: _largest(model_rows, f"{split}_gap_audio_minus_decision") for split in ("speaker", "statement")
        },
        "largest_gap_with_audio_probe_at_least_0_9": {
            split: _largest(
                model_rows,
                f"{split}_gap_audio_minus_decision",
                lambda row: row[f"audio_{split}_probe"] >= 0.9,
            )
            for split in ("speaker", "statement")
        },
        "interpretation": "Probe gap is a layerwise readout-retention diagnostic, not a causal proof of a single transfer bottleneck.",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-accuracy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_transfer_rows(_read_probe_rows(args.probe_accuracy))
    summary = {"source": str(args.probe_accuracy), **summarize_transfer(rows)}
    _write_csv(args.output_dir / "decision_transfer.csv", rows)
    (args.output_dir / "decision_transfer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        for column, title, axis in (
            ("speaker", "Speaker-held-out probe", axes[0, 0]),
            ("statement", "Statement-held-out probe", axes[0, 1]),
        ):
            axis.plot([row["layer_index"] for row in rows], [row[f"audio_{column}_probe"] for row in rows], marker=".", label="audio-token mean")
            axis.plot([row["layer_index"] for row in rows], [row[f"decision_{column}_probe"] for row in rows], marker=".", label="decision state")
            axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1, label="chance")
            axis.set(title=title, xlabel="LLM layer index", ylabel="Accuracy")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes[1, 0].plot(
            [row["layer_index"] for row in rows],
            [row["speaker_gap_audio_minus_decision"] for row in rows],
            marker=".",
            label="speaker split",
        )
        axes[1, 0].plot(
            [row["layer_index"] for row in rows],
            [row["statement_gap_audio_minus_decision"] for row in rows],
            marker=".",
            label="statement split",
        )
        axes[1, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        axes[1, 0].set(title="Audio minus decision probe gap", xlabel="LLM layer index", ylabel="Accuracy difference")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].plot(
            [row["layer_index"] for row in rows],
            [row["speaker_decision_step"] if row["speaker_decision_step"] is not None else float("nan") for row in rows],
            marker=".",
            label="speaker split",
        )
        axes[1, 1].plot(
            [row["layer_index"] for row in rows],
            [row["statement_decision_step"] if row["statement_decision_step"] is not None else float("nan") for row in rows],
            marker=".",
            label="statement split",
        )
        axes[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        axes[1, 1].set(title="Decision-state layer step", xlabel="LLM layer index", ylabel="Accuracy change")
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(fontsize=8)
        fig.savefig(args.output_dir / "decision_transfer.png", dpi=180)
        plt.close(fig)
    except ImportError:
        (args.output_dir / "plot_unavailable.txt").write_text("matplotlib is not installed; CSV and JSON outputs remain available.\n")


if __name__ == "__main__":
    main()
