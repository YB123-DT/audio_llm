#!/usr/bin/env python3
"""Summarize teacher-forced forced-choice likelihood diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    required = {
        "sample_id",
        "pair_id",
        "actor",
        "statement",
        "repetition",
        "intensity",
        "emotion",
        "ground_truth",
        "condition_id",
        "prompt",
        "happy_verbalizer",
        "sad_verbalizer",
        "happy_token_ids",
        "sad_token_ids",
        "happy_logprob",
        "sad_logprob",
        "score_happy_minus_sad",
        "predicted",
        "correct",
        "error",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Forced-choice CSVs are missing required columns: {sorted(required)}")
    return rows


def _pair_margins(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["emotion"]] = row
    margins: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        if set(pair) != {"happy", "sad"}:
            raise ValueError(f"Pair {pair_id} does not contain exactly happy and sad rows")
        happy = pair["happy"]
        sad = pair["sad"]
        for key in ("actor", "statement", "repetition", "intensity"):
            if happy[key] != sad[key]:
                raise ValueError(f"Pair {pair_id} metadata mismatch for {key}")
        margin = float(happy["score_happy_minus_sad"]) - float(sad["score_happy_minus_sad"])
        margins.append(
            {
                "pair_id": pair_id,
                "actor": happy["actor"],
                "statement": happy["statement"],
                "repetition": happy["repetition"],
                "intensity": happy["intensity"],
                "happy_score": float(happy["score_happy_minus_sad"]),
                "sad_score": float(sad["score_happy_minus_sad"]),
                "pair_margin_happy_minus_sad": margin,
                "pair_direction_positive": margin > 0,
                "pair_direction_tie": margin == 0,
            }
        )
    return margins


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def summarize_condition(condition_id: str, rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != 192:
        raise ValueError(f"Condition {condition_id} has {len(rows)} rows; expected 192")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Condition {condition_id} contains duplicate sample IDs")
    errors = [row for row in rows if row["error"]]
    if errors:
        raise ValueError(f"Condition {condition_id} contains {len(errors)} runtime errors")
    labels = [1 if row["emotion"] == "happy" else 0 for row in rows]
    scores = [float(row["score_happy_minus_sad"]) for row in rows]
    correct = sum(row["correct"] == "True" for row in rows)
    predictions = Counter(row["predicted"] for row in rows)
    auc = _roc_auc(labels, scores)
    margins = _pair_margins(rows)
    margin_values = [row["pair_margin_happy_minus_sad"] for row in margins]
    positive = sum(value > 0 for value in margin_values)
    ties = sum(value == 0 for value in margin_values)
    first = rows[0]
    summary = {
        "condition_id": condition_id,
        "prompt": first["prompt"],
        "happy_verbalizer": first["happy_verbalizer"],
        "sad_verbalizer": first["sad_verbalizer"],
        "happy_token_ids": first["happy_token_ids"],
        "sad_token_ids": first["sad_token_ids"],
        "n_rows": len(rows),
        "n_pairs": len(margins),
        "forced_choice_accuracy": correct / len(rows),
        "correct_count": correct,
        "predicted_counts": dict(predictions),
        "roc_auc": auc,
        "pair_direction_positive_fraction": positive / len(margins) if margins else None,
        "pair_direction_tie_count": ties,
        "pair_margin_mean": sum(margin_values) / len(margin_values) if margin_values else None,
        "pair_margin_min": min(margin_values) if margin_values else None,
        "pair_margin_max": max(margin_values) if margin_values else None,
    }
    return summary, [{"condition_id": condition_id, **row} for row in margins]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": summary["condition_id"],
        "prompt": summary["prompt"],
        "happy_verbalizer": summary["happy_verbalizer"],
        "sad_verbalizer": summary["sad_verbalizer"],
        "happy_token_ids": summary["happy_token_ids"],
        "sad_token_ids": summary["sad_token_ids"],
        "n_rows": summary["n_rows"],
        "n_pairs": summary["n_pairs"],
        "forced_choice_accuracy": summary["forced_choice_accuracy"],
        "correct_count": summary["correct_count"],
        "roc_auc": summary["roc_auc"],
        "pair_direction_positive_fraction": summary["pair_direction_positive_fraction"],
        "pair_direction_tie_count": summary["pair_direction_tie_count"],
        "pair_margin_mean": summary["pair_margin_mean"],
        "pair_margin_min": summary["pair_margin_min"],
        "pair_margin_max": summary["pair_margin_max"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(args.predictions)
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for condition_id in sorted(by_condition):
        summary, margins = summarize_condition(condition_id, by_condition[condition_id])
        summaries.append(summary)
        pair_rows.extend(margins)
    _write_csv(args.output_dir / "forced_choice_summary.csv", [_summary_csv_row(summary) for summary in summaries])
    _write_csv(args.output_dir / "forced_choice_pair_margins.csv", pair_rows)
    (args.output_dir / "forced_choice_summary.json").write_text(json.dumps({"conditions": summaries}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
