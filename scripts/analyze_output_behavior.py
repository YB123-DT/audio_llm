#!/usr/bin/env python3
"""Summarize free-generation sensitivity to the emotion swap in matched pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PAIR_FIELDS = ("pair_id", "actor", "statement", "repetition", "intensity")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "pair_id", "actor", "statement", "repetition", "intensity", "emotion", "model_output"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing required prediction columns: {sorted(required)}")
    return rows


def _pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], {})[row["emotion"]] = row
    pair_rows: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        if set(pair) != {"happy", "sad"}:
            raise ValueError(f"Pair {pair_id} does not contain exactly happy and sad rows")
        happy = pair["happy"]
        sad = pair["sad"]
        if any(happy[field] != sad[field] for field in ("actor", "statement", "repetition", "intensity")):
            raise ValueError(f"Pair {pair_id} metadata does not match across emotions")
        pair_rows.append(
            {
                **{field: happy[field] for field in PAIR_FIELDS},
                "happy_output": happy["model_output"],
                "sad_output": sad["model_output"],
                "changed": happy["model_output"] != sad["model_output"],
            }
        )
    return pair_rows


def _mode_stats(rows: list[dict[str, str]], statement: str) -> dict[str, Any]:
    outputs = [row["model_output"] for row in rows if row["statement"] == statement]
    counts = Counter(outputs)
    most_common_output, most_common_count = counts.most_common(1)[0] if counts else ("", 0)
    return {
        "n_rows": len(outputs),
        "unique_outputs": len(counts),
        "mode_count": most_common_count,
        "mode_fraction": most_common_count / len(outputs) if outputs else None,
        "mode_output": most_common_output,
    }


def summarize_prediction_file(label: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_rows(path)
    pairs = _pair_rows(rows)
    if len(rows) != 2 * len(pairs):
        raise ValueError(f"{path} has duplicate or incomplete pair rows")
    changed = [pair for pair in pairs if pair["changed"]]
    summary = {
        "condition": label,
        "source": str(path),
        "n_rows": len(rows),
        "n_pairs": len(pairs),
        "same_output_pairs": len(pairs) - len(changed),
        "changed_output_pairs": len(changed),
        "output_sensitivity": len(changed) / len(pairs) if pairs else None,
        "statement_stats": {statement: _mode_stats(rows, statement) for statement in ("01", "02")},
        "changed_pair_ids": [pair["pair_id"] for pair in changed],
    }
    return summary, [{"condition": label, **pair} for pair in pairs]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Prediction file, repeatable; e.g. max8=reports/.../predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for spec in args.prediction:
        if "=" not in spec:
            parser.error(f"--prediction must use LABEL=CSV: {spec}")
        label, filename = spec.split("=", 1)
        if not label or not filename:
            parser.error(f"--prediction must use LABEL=CSV: {spec}")
        summary, pairs = summarize_prediction_file(label, Path(filename))
        summaries.append(summary)
        pair_rows.extend(pairs)

    _write_csv(args.output_dir / "output_sensitivity.csv", summaries)
    _write_csv(args.output_dir / "output_sensitivity_pairs.csv", pair_rows)
    (args.output_dir / "output_sensitivity_summary.json").write_text(json.dumps({"conditions": summaries}, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
