#!/usr/bin/env python3
"""Analyze pooled SLAM-Omni representations with matched-pair PS and probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _cosine_parallelism(deltas: np.ndarray) -> float:
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    normalized = deltas / np.maximum(norms, 1e-12)
    sims = normalized @ normalized.T
    mask = ~np.eye(len(deltas), dtype=bool)
    return float(sims[mask].mean()) if mask.any() else float("nan")


def _pair_deltas(features: np.ndarray, metadata: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    grouped: dict[str, dict[str, tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(metadata):
        grouped.setdefault(row["pair_id"], {})[row["emotion"]] = (index, row)
    deltas: list[np.ndarray] = []
    pair_rows: list[dict[str, str]] = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        if set(pair) != {"happy", "sad"}:
            continue
        sad_index, sad_row = pair["sad"]
        happy_index, _ = pair["happy"]
        deltas.append(features[sad_index] - features[happy_index])
        pair_rows.append({"pair_id": pair_id, **{k: sad_row[k] for k in ("actor", "statement", "repetition", "intensity")}})
    if not deltas:
        raise ValueError("No complete happy/sad pairs in hidden metadata")
    return np.stack(deltas), pair_rows


def _probe_accuracy(features: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(labels[train])) < 2:
        return float("nan")
    classifier = LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)
    classifier.fit(features[train], labels[train])
    return float((classifier.predict(features[test]) == labels[test]).mean())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    payload = torch.load(args.representations, map_location="cpu", weights_only=False)
    with args.metadata.open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    labels = np.asarray([1 if row["emotion"] == "happy" else 0 for row in metadata])
    actors = np.asarray([row["actor"] for row in metadata])
    statements = np.asarray([row["statement"] for row in metadata])
    held_out_actors = sorted(set(actors))[-6:]
    speaker_test = np.isin(actors, held_out_actors)
    speaker_train = ~speaker_test

    feature_sets: list[tuple[str, np.ndarray, list[str]]] = []
    audio_names = [str(x) for x in payload["audio_layer_names"]]
    feature_sets.append(("audio_encoder", payload["audio_encoder"].numpy(), audio_names))
    feature_sets.append(("projector", payload["projector"].numpy()[:, None, :], ["projector_mean"]))
    llm_names = [str(x) for x in payload["llm_layer_names"]]
    feature_sets.append(("llm_audio_mean", payload["llm_audio_mean"].numpy(), llm_names))
    feature_sets.append(("llm_decision", payload["llm_decision"].numpy(), llm_names))
    feature_sets.append(("llm_prompt_last", payload["llm_prompt_last"].numpy(), llm_names))

    parallel_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for family, tensor, names in feature_sets:
        for layer_index, layer_name in enumerate(names):
            features = tensor[:, layer_index, :]
            deltas, pair_rows = _pair_deltas(features, metadata)
            parallel_rows.append({"family": family, "layer_index": layer_index, "layer": layer_name, "pairs": len(pair_rows), "parallelism_cosine": _cosine_parallelism(deltas)})

            statement_results = []
            for train_statement, test_statement in (("01", "02"), ("02", "01")):
                train_mask = statements == train_statement
                test_mask = statements == test_statement
                statement_results.append(_probe_accuracy(features, labels, train_mask, test_mask))
            probe_rows.append(
                {
                    "family": family,
                    "layer_index": layer_index,
                    "layer": layer_name,
                    "speaker_held_out_actors": ",".join(held_out_actors),
                    "speaker_held_out_accuracy": _probe_accuracy(features, labels, speaker_train, speaker_test),
                    "statement_01_to_02_accuracy": statement_results[0],
                    "statement_02_to_01_accuracy": statement_results[1],
                    "statement_held_out_mean": float(np.nanmean(statement_results)),
                }
            )

    _write_csv(args.output_dir / "parallelism.csv", parallel_rows)
    _write_csv(args.output_dir / "probe_accuracy.csv", probe_rows)
    summary = {
        "n_rows": len(metadata),
        "n_pairs": len(_pair_deltas(payload["projector"].numpy(), metadata)[1]),
        "actors": sorted(set(actors)),
        "held_out_actors": held_out_actors,
        "prompt": payload.get("prompt"),
        "feature_families": [family for family, _, _ in feature_sets],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        for family, _, names in feature_sets:
            selected = [r for r in parallel_rows if r["family"] == family]
            axes[0].plot([r["layer_index"] for r in selected], [r["parallelism_cosine"] for r in selected], marker=".", label=family)
            selected_probe = [r for r in probe_rows if r["family"] == family]
            axes[1].plot([r["layer_index"] for r in selected_probe], [r["speaker_held_out_accuracy"] for r in selected_probe], marker=".", label=family)
        axes[0].set(title="Matched sad-minus-happy parallelism", xlabel="Layer index", ylabel="Mean off-diagonal cosine")
        axes[1].set(title="Speaker-held-out emotion probe", xlabel="Layer index", ylabel="Accuracy")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.savefig(args.output_dir / "diagnostic_curves.png", dpi=180)
        plt.close(fig)
    except ImportError:
        (args.output_dir / "plot_unavailable.txt").write_text("matplotlib is not installed; CSV outputs remain available.\n")


if __name__ == "__main__":
    main()
