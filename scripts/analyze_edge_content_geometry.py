#!/usr/bin/env python3
"""Descriptive statement-conditioned clean edge geometry, with actor uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def raw_coefficients(coef, intercept, mean, scale):
    w = np.asarray(coef).reshape(-1) / scale
    return w, float(np.asarray(intercept).reshape(-1)[0] - w @ mean)


def geometry(means):
    """Means shape [..., statement(01,02), emotion(happy,sad), dimension]."""
    d1 = means[..., 0, 0, :] - means[..., 0, 1, :]
    d2 = means[..., 1, 0, :] - means[..., 1, 1, :]
    offset = means[..., 1, :, :].mean(axis=-2) - means[..., 0, :, :].mean(axis=-2)
    shared = (d1 + d2) / 2
    norm = lambda v: np.linalg.norm(v, axis=-1)
    dot = lambda a, b: np.sum(a * b, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = {'cos_emotion_directions': dot(d1, d2) / (norm(d1) * norm(d2)),
                  'norm_offset': norm(offset), 'norm_d01': norm(d1), 'norm_d02': norm(d2),
                  'norm_dshared': norm(shared)}
        for name, direction in [('d01', d1), ('d02', d2), ('dshared', shared)]:
            result['offset_dot_' + name] = dot(offset, direction)
            result['offset_projection_' + name] = dot(offset, direction) / norm(direction)
            result['offset_to_gap_' + name] = dot(offset, direction) / dot(direction, direction)
            result['offset_cos_' + name] = dot(offset, direction) / (norm(offset) * norm(direction))
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze(representations, metadata, output, bootstrap=10000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sklearn

    data = np.load(representations, allow_pickle=False)
    x = data['edge_updates'] if 'edge_updates' in data else data['edge_vectors']
    rows = list(csv.DictReader(metadata.open()))
    lookup = {r['sample_id']: r for r in rows}
    ids = [str(v) for v in data['sample_ids']]
    if len(lookup) != len(rows) or len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError('Metadata/representation IDs must match uniquely')
    rows = [lookup[s] for s in ids]
    actors = sorted({r['actor'] for r in rows})
    statements = sorted({r['statement'] for r in rows})
    if statements != ['01', '02'] or {r['emotion'] for r in rows} != {'happy', 'sad'}:
        raise ValueError('Expected two statements and happy/sad')
    if x.shape[:2] != (len(rows), len(data['layers'])) or not np.isfinite(x).all():
        raise ValueError('Invalid representation tensor')
    actor_means = []
    cell_sizes = []
    for actor in actors:
        cells = []
        for statement in statements:
            emotions = []
            for emotion in ['happy', 'sad']:
                indices = [i for i, r in enumerate(rows) if (r['actor'], r['statement'], r['emotion']) == (actor, statement, emotion)]
                if not indices:
                    raise ValueError('Incomplete actor/statement/emotion cells')
                cell_sizes.append(len(indices))
                emotions.append(x[indices].astype(np.float64).mean(axis=0))
            cells.append(emotions)
        actor_means.append(cells)
    if len(set(cell_sizes)) != 1:
        raise ValueError('Expected equal repetitions per actor/statement/emotion cell')
    # actor, statement, emotion, layer, dimension
    actor_means = np.asarray(actor_means)
    means = actor_means.mean(axis=0).transpose(2, 0, 1, 3)
    rng = np.random.default_rng(20260915)
    draws = rng.integers(len(actors), size=(bootstrap, len(actors)))
    counts = np.array([np.bincount(draw, minlength=len(actors)) for draw in draws]) / len(actors)
    summaries, probe_rows, all_weights, all_intercepts = [], [], [], []
    max_parity = 0.
    for li, layer in enumerate(data['layers']):
        point = geometry(means[li])
        boot = {key: [] for key in point}
        for start in range(0, bootstrap, 128):
            sampled = np.einsum('ba,ased->bsed', counts[start:start + 128], actor_means[:, :, :, li, :], optimize=True)
            for key, value in geometry(sampled).items():
                boot[key].append(value)
        for metric, value in point.items():
            distribution = np.concatenate(boot[metric])
            lo, hi = np.quantile(distribution, [.025, .975])
            summaries.append(dict(layer=int(layer), metric=metric, mean=float(value), ci_low=float(lo), ci_high=float(hi)))
        fitted = []
        for statement in statements:
            indices = np.array([i for i, r in enumerate(rows) if r['statement'] == statement])
            xx = x[indices, li].astype(np.float64)
            yy = np.array([rows[i]['emotion'] == 'happy' for i in indices])
            scaler = StandardScaler().fit(xx)
            model = LogisticRegression(C=1., solver='lbfgs', max_iter=5000, random_state=20260915).fit(scaler.transform(xx), yy)
            if model.n_iter_.max() >= 5000:
                raise RuntimeError('Probe failed to converge')
            w, intercept = raw_coefficients(model.coef_, model.intercept_, scaler.mean_, scaler.scale_)
            parity = float(np.max(np.abs(xx @ w + intercept - model.decision_function(scaler.transform(xx)))))
            max_parity = max(max_parity, parity)
            fitted.append((w, intercept))
        d1 = means[li, 0, 0] - means[li, 0, 1]
        d2 = means[li, 1, 0] - means[li, 1, 1]
        offset = means[li, 1].mean(axis=0) - means[li, 0].mean(axis=0)
        (w1, b1), (w2, b2) = fitted
        record = dict(layer=int(layer), cosine_probe_directions=float(w1 @ w2 / np.linalg.norm(w1) / np.linalg.norm(w2)))
        for statement, (w, b) in zip(statements, fitted):
            norm = np.linalg.norm(w)
            record.update({f'w{statement}_norm': float(norm), f'b{statement}_raw': b,
                           f'b{statement}_normalized': float(b / norm), f'w{statement}_offset': float(w @ offset),
                           f'w{statement}_offset_normalized': float(w @ offset / norm),
                           f'w{statement}_gap01': float(w @ d1), f'w{statement}_gap02': float(w @ d2)})
        probe_rows.append(record)
        all_weights.append([w1, w2])
        all_intercepts.append([b1, b2])
    if max_parity > 1e-8:
        raise ValueError('Raw coefficient score parity failed')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'geometry_summary.csv', summaries)
    write_csv(output / 'geometry_probe_directions.csv', probe_rows)
    np.savez_compressed(output / 'geometry_vectors.npz', class_means=means, layers=data['layers'], statements=np.array(statements), emotions=np.array(['happy', 'sad']), probe_weights=np.asarray(all_weights), probe_intercepts=np.asarray(all_intercepts), d01=means[:, 0, 0]-means[:, 0, 1], d02=means[:, 1, 0]-means[:, 1, 1], offset=means[:, 1].mean(axis=1)-means[:, 0].mean(axis=1))
    details = dict(samples=len(rows), actors=len(actors), repetitions_per_cell=cell_sizes[0], bootstrap=bootstrap, seed=20260915, input_sha256=hashlib.sha256(representations.read_bytes()).hexdigest(), sklearn_version=sklearn.__version__, raw_coordinate_parity_max=max_parity, uncertainty='Unadjusted percentile 95% actor-cluster bootstrap of class-mean geometry; nonlinear high-dimensional norms, cosines and ratios can have biased bootstrap distributions, so intervals are descriptive and are not bias-corrected hypothesis tests; independently fitted statement probe comparisons are descriptive only, no bootstrap/refit claims.', caveats='Per-statement probes use all examples of that statement for descriptive direction comparison, not held-out accuracy. Raw and norm-normalized intercept differences do not identify shared-threshold shifts unless directions align. High dimensional regularized directions are not unique. Offset projection uses happy-minus-sad axes; offset-to-gap = o dot d / ||d||^2.')
    (output / 'geometry_analysis.json').write_text(json.dumps(details, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representations', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    analyze(args.representations, args.metadata, args.output_dir, args.bootstrap)
