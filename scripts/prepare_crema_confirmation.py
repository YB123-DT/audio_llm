#!/usr/bin/env python3
"""Freeze complete CREMA-D XX happy/sad cells before independent confirmation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import wave

REVISION = '1658cd342dff90010aa843eaeebd53610a08b1dc'
REPO = 'CheyneyComputerScience/CREMA-D'
STATEMENTS = ('DFA', 'IOM', 'ITH', 'ITS', 'IWL', 'IWW', 'MTI', 'TAI', 'TIE', 'TSI', 'WSI')


def selection(tree):
    if tree.get('sha') != REVISION or tree.get('truncated'):
        raise ValueError('Unexpected or truncated upstream tree')
    cells = {}
    for item in tree['tree']:
        path = item['path']
        if not path.startswith('AudioWAV/') or not path.endswith('.wav'):
            continue
        actor, statement, emotion, intensity = Path(path).stem.split('_')
        if intensity != 'XX' or emotion not in ('HAP', 'SAD'):
            continue
        key = actor, statement, emotion
        if key in cells:
            raise ValueError('Duplicate cell')
        cells[key] = path
    statements = sorted({k[1] for k in cells})
    if tuple(statements) != STATEMENTS:
        raise ValueError('Unexpected sentence coverage')
    actors = sorted({k[0] for k in cells})
    missing = {a: [s for s in statements if any((a, s, e) not in cells for e in ('HAP', 'SAD'))] for a in actors}
    included = [a for a in actors if not missing[a]]
    rows = []
    for actor in included:
        for statement in statements:
            pair = f'{actor}_{statement}_XX'
            for code, label in [('HAP', 'happy'), ('SAD', 'sad')]:
                rows.append(dict(sample_id=f'{pair}_{label}', pair_id=pair, path=cells[actor, statement, code], actor=actor, statement=statement, repetition='01', intensity='XX', emotion=label))
    return rows, {a: s for a, s in missing.items() if s}


def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def validate_audio(raw):
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        n, rate = wav.getnframes(), wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or rate != 16000 or n <= 0:
            raise ValueError('Unexpected CREMA WAV encoding')
        if len(wav.readframes(n)) != n * 2:
            raise ValueError('Truncated WAV')
        return dict(sample_rate=rate, frames=n, duration_seconds=n/rate, sha256=hashlib.sha256(raw).hexdigest())


def download(row, root):
    dest = root / row['path']
    raw = dest.read_bytes() if dest.exists() else fetch(f'https://media.githubusercontent.com/media/{REPO}/{REVISION}/{row["path"]}')
    info = validate_audio(raw)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.part')
        tmp.write_bytes(raw)
        tmp.replace(dest)
    return dict(sample_id=row['sample_id'], path=row['path'], **info)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree-cache', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--audio-root', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    tree_raw = args.tree_cache.read_bytes() if args.tree_cache else fetch(f'https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1')
    tree = json.loads(tree_raw)
    rows, excluded = selection(tree)
    if len(rows) != 1936 or len({r['actor'] for r in rows}) != 88:
        raise ValueError('Pinned source completeness changed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir/'manifest.csv'
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    content = buf.getvalue()
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError('Refuse to change frozen manifest')
    manifest.write_text(content)
    provenance = dict(source=f'https://github.com/{REPO}', revision=REVISION, tree_sha256=hashlib.sha256(tree_raw).hexdigest(), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), n_samples=len(rows), n_pairs=len(rows)//2, n_actors=88, statements=STATEMENTS, excluded_actors=excluded, selection='All actors with complete happy/sad pairs for all eleven XX statements; selected only from filenames before model outputs. No quality/rating/performance-based exclusions.', intensity='XX means unspecified; it is NOT a controlled equal acoustic intensity.', repetition='01 is an interface placeholder: CREMA-D provides one file per selected actor/statement/emotion cell, not repeated takes.', label='Intended acted emotion encoded in filenames; not crowd-vote relabeling.', source_actor_identity='Separate corpus and source actor IDs; cross-corpus real-person identity overlap is not independently established.')
    (args.output_dir/'selection.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2), flush=True)
    if args.download:
        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download, row, args.audio_root) for row in rows]
            for f in as_completed(futures):
                records.append(f.result())
                if len(records) % 100 == 0:
                    print(f'Downloaded/validated {len(records)}/{len(rows)}', flush=True)
        lookup = {r['sample_id']: r for r in records}
        with (args.output_dir/'audio_checksums.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator='\n'); writer.writeheader()
            writer.writerows(lookup[r['sample_id']] for r in rows)
        print('All selected WAV files validated.', flush=True)


if __name__ == '__main__':
    main()
