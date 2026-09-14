#!/usr/bin/env python3
"""Build a strictly matched happy/sad RAVDESS speech manifest.

RAVDESS filenames encode seven fields:
modality-channel-emotion-intensity-statement-repetition-actor.wav

The first diagnostic intentionally uses audio-only speech (03-01), one fixed
intensity, and keeps only pairs that differ in emotion (03=happy, 04=sad).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


EMOTION_NAMES = {"03": "happy", "04": "sad"}
REQUIRED_COLUMNS = [
    "sample_id",
    "pair_id",
    "path",
    "actor",
    "statement",
    "repetition",
    "intensity",
    "emotion",
]


@dataclass(frozen=True)
class Clip:
    path: Path
    actor: str
    statement: str
    repetition: str
    intensity: str
    emotion: str

    @property
    def pair_key(self) -> tuple[str, str, str, str]:
        return self.actor, self.statement, self.repetition, self.intensity


def parse_clip(path: Path, ravdess_root: Path) -> Clip | None:
    fields = path.stem.split("-")
    if len(fields) != 7 or fields[0] != "03" or fields[1] != "01":
        return None
    emotion = EMOTION_NAMES.get(fields[2])
    if emotion is None:
        return None
    if fields[-1] != path.parent.name.removeprefix("Actor_"):
        raise ValueError(f"Actor field disagrees with directory: {path}")
    return Clip(
        path=path.relative_to(ravdess_root),
        actor=fields[-1],
        statement=fields[4],
        repetition=fields[5],
        intensity=fields[3],
        emotion=emotion,
    )


def build_rows(ravdess_root: Path, intensity: str) -> list[dict[str, str]]:
    clips: list[Clip] = []
    for path in sorted(ravdess_root.rglob("*.wav")):
        clip = parse_clip(path, ravdess_root)
        if clip is not None and (intensity == "all" or clip.intensity == intensity):
            clips.append(clip)

    by_key: dict[tuple[str, str, str, str], dict[str, Clip]] = {}
    for clip in clips:
        by_key.setdefault(clip.pair_key, {})[clip.emotion] = clip

    rows: list[dict[str, str]] = []
    pair_index = 0
    for pair_key in sorted(by_key):
        pair = by_key[pair_key]
        if set(pair) != {"happy", "sad"}:
            continue
        pair_id = f"pair_{pair_index:04d}"
        pair_index += 1
        for emotion in ("happy", "sad"):
            clip = pair[emotion]
            sample_id = f"{pair_id}_{emotion}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "path": clip.path.as_posix(),
                    "actor": clip.actor,
                    "statement": clip.statement,
                    "repetition": clip.repetition,
                    "intensity": clip.intensity,
                    "emotion": clip.emotion,
                }
            )
    return rows


def validate_rows(rows: list[dict[str, str]], ravdess_root: Path, intensity: str) -> None:
    if not rows:
        raise ValueError("Manifest is empty")
    if list(rows[0]) != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(rows[0])}")
    if len(rows) % 2:
        raise ValueError("Manifest must contain two rows per matched pair")
    seen_paths: set[str] = set()
    by_pair: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["path"] in seen_paths:
            raise ValueError(f"Duplicate path: {row['path']}")
        seen_paths.add(row["path"])
        if not (ravdess_root / row["path"]).is_file():
            raise ValueError(f"Missing audio: {ravdess_root / row['path']}")
        if row["intensity"] != intensity:
            raise ValueError(f"Unexpected intensity in row: {row}")
        by_pair.setdefault(row["pair_id"], []).append(row)

    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2 or {r["emotion"] for r in pair_rows} != {"happy", "sad"}:
            raise ValueError(f"Pair {pair_id} is not exactly happy/sad")
        keys = {(r["actor"], r["statement"], r["repetition"], r["intensity"]) for r in pair_rows}
        if len(keys) != 1:
            raise ValueError(f"Pair {pair_id} metadata differs: {keys}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravdess-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intensity", choices=["01", "02", "all"], default="01")
    args = parser.parse_args()
    root = args.ravdess_root.resolve()
    rows = build_rows(root, args.intensity)
    validate_rows(rows, root, args.intensity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(rows) // 2} pairs) to {args.output}")


if __name__ == "__main__":
    main()
