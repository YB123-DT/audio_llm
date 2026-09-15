#!/usr/bin/env python3
"""Out-of-fold clean causal-edge probes and coordinate-correct direction geometry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def folds(rows, regime):
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    for actor in (sorted(set(actors)) if regime != 'statement' else [None]):
        for statement in (sorted(set(statements)) if regime != 'speaker' else [None]):
            train = np.ones(len(rows), dtype=bool)
            test = train.copy()
            if actor is not None:
                train &= actors != actor
                test &= actors == actor
            if statement is not None:
                train &= statements != statement
                test &= statements == statement
            yield f'actor_{actor}_statement_{statement}', np.flatnonzero(train), np.flatnonzero(test)


def raw_direction(coef, scale):
    return np.asarray(coef).reshape(-1) / np.asarray(scale)


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else float('nan')


def auc(y, score):
    h, s = score[y == 1], score[y == 0]
    return float(((h[:, None] > s).astype(float) + .5 * (h[:, None] == s)).mean())


def clustered_metrics(y, score, actor, counts):
    """Fixed-fold OOF uncertainty; actor multiplicity preserves cross-actor AUC."""
    unique = sorted(set(actor))
    idx = [np.flatnonzero(actor == a) for a in unique]
    sizes = np.array([len(i) for i in idx])
    correct = np.array([((score[i] > 0) == y[i]).sum() for i in idx])
    h = [score[i][y[i] == 1] for i in idx]
    s = [score[i][y[i] == 0] for i in idx]
    kernel = np.array([[((hi[:, None] > sj).astype(float) + .5 * (hi[:, None] == sj)).sum() for sj in s] for hi in h])
    boots_auc = np.einsum('bi,ij,bj->b', counts, kernel, counts) / ((counts @ np.array([len(v) for v in h])) * (counts @ np.array([len(v) for v in s])))
    boots_acc = counts @ correct / (counts @ sizes)
    result = []
    for name, mean, boots in [('accuracy', float(((score > 0) == y).mean()), boots_acc), ('roc_auc', auc(y, score), boots_auc)]:
        low, high = np.quantile(boots, [.025, .975])
        result.append(dict(metric=name, mean=mean, ci_low=float(low), ci_high=float(high)))
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze(npz_path, metadata, out, bootstrap=10000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sklearn

    data = np.load(npz_path, allow_pickle=False)
    vectors = data['edge_updates'] if 'edge_updates' in data else data['edge_vectors']
    layer_ids = data['layers']
    source_rows = list(csv.DictReader(Path(metadata).open()))
    lookup = {r['sample_id']: r for r in source_rows}
    if len(lookup) != len(source_rows):
        raise ValueError('Duplicate metadata sample IDs')
    rows = [lookup[str(s)] for s in data['sample_ids']]
    if len(rows) != len(lookup) or len(set(data['sample_ids'])) != len(rows):
        raise ValueError('Representation and metadata sample IDs differ')
    if vectors.shape[:2] != (len(rows), len(layer_ids)) or not np.isfinite(vectors).all():
        raise ValueError('Invalid edge vectors')
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    if {r['emotion'] for r in rows} != {'happy', 'sad'}:
        raise ValueError('Expected happy/sad labels')
    actors = np.array([r['actor'] for r in rows])
    unique = sorted(set(actors))
    draws = np.random.default_rng(20260915).integers(0, len(unique), (bootstrap, len(unique)))
    counts = np.array([np.bincount(d, minlength=len(unique)) for d in draws])
    axes = {'lm_raw': data['d_lm']}
    for key in ['d_lm_norm_weighted', 'd_norm']:
        if key in data:
            axes['lm_norm_weighted'] = data[key]
            break
    summaries, predictions, fold_results, directions = [], [], [], []
    weights, weight_names = [], []
    for li, layer in enumerate(layer_ids):
        x = vectors[:, li].astype(np.float64)
        for regime in ['speaker', 'statement', 'joint']:
            oof = np.full(len(y), np.nan)
            coverage = np.zeros(len(y), dtype=int)
            for fold, train, test in folds(rows, regime):
                scaler = StandardScaler().fit(x[train])
                clf = LogisticRegression(C=1., max_iter=5000, solver='lbfgs', random_state=20260915)
                clf.fit(scaler.transform(x[train]), y[train])
                if clf.n_iter_.max() >= 5000:
                    raise RuntimeError('Probe failed to converge')
                scores = clf.decision_function(scaler.transform(x[test]))
                oof[test] = scores
                coverage[test] += 1
                w = raw_direction(clf.coef_, scaler.scale_)
                intercept = float(clf.intercept_[0] - np.dot(w, scaler.mean_))
                parity = float(np.max(np.abs(x[test] @ w + intercept - scores)))
                if parity > 1e-8:
                    raise ValueError('Raw-coordinate probe parity failure')
                fold_results.append(dict(layer=int(layer), regime=regime, fold=fold, n_train=len(train), n_test=len(test), accuracy=float(((scores > 0) == y[test]).mean()), roc_auc=auc(y[test], scores), raw_score_parity_max=parity))
                for axis, direction in axes.items():
                    axis_score = x[test] @ direction
                    correlation = float(np.corrcoef(scores, axis_score)[0, 1]) if np.std(scores) > 0 and np.std(axis_score) > 0 else float('nan')
                    directions.append(dict(layer=int(layer), regime=regime, fold=fold, axis=axis, cosine=cosine(w, direction), test_score_correlation=correlation, probe_norm=float(np.linalg.norm(w)), axis_norm=float(np.linalg.norm(direction))))
                weights.append(w)
                weight_names.append(f'{int(layer)}:{regime}:{fold}')
                for i, score in zip(test, scores):
                    predictions.append(dict(layer=int(layer), regime=regime, fold=fold, sample_id=rows[i]['sample_id'], actor=rows[i]['actor'], statement=rows[i]['statement'], emotion=rows[i]['emotion'], score=float(score), prediction='happy' if score > 0 else 'sad'))
            if not np.all(coverage == 1) or not np.isfinite(oof).all():
                raise ValueError('Every sample must receive exactly one held-out prediction')
            for metric in clustered_metrics(y, oof, actors, counts):
                summaries.append(dict(layer=int(layer), readout=regime, **metric))
            fold_aucs = [r['roc_auc'] for r in fold_results if r['layer'] == int(layer) and r['regime'] == regime]
            summaries.append(dict(layer=int(layer), readout=regime, metric='mean_within_fold_auc', mean=float(np.mean(fold_aucs)), ci_low='', ci_high=''))
        for axis, direction in axes.items():
            score = x @ direction
            for metric in clustered_metrics(y, score, actors, counts):
                # An edge-only dot product has no calibrated classification threshold.
                if metric['metric'] == 'roc_auc':
                    summaries.append(dict(layer=int(layer), readout=axis, **metric))
    if 'clean_margins' in data:
        for metric in clustered_metrics(y, data['clean_margins'], actors, counts):
            summaries.append(dict(layer='final', readout='model_clean', **metric))
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [('probe_summary', summaries), ('oof_predictions', predictions), ('fold_metrics', fold_results), ('direction_cosines', directions)]:
        write_csv(out / f'{name}.csv', values)
    np.savez_compressed(out / 'probe_directions.npz', weights=np.stack(weights), names=np.array(weight_names))
    details = dict(n_samples=len(rows), n_actors=len(unique), layers=layer_ids.tolist(), input_sha256=hashlib.sha256(Path(npz_path).read_bytes()).hexdigest(), sklearn_version=sklearn.__version__, bootstrap=bootstrap, seed=20260915, uncertainty='Actor-cluster percentile 95% conditional on trained folds; models are not refit in bootstrap; unadjusted intervals.', preprocessing='Train-only StandardScaler; logistic regression C=1 lbfgs max_iter=5000; happy positive; no tuning.', geometry='w_raw=coef/scale; cosine with raw or final-RMSNorm-weighted unembedding difference is descriptive. Early edge updates undergo downstream computation; final RMS denominator is state dependent. Near-zero cosine alone does not prove readout failure.', comparability='Train-only standardization and leave-one-actor-out differ from earlier unstandardized fixed speaker split; scores are not numerically matched protocols. Dimension896 exceeds per-fold training count, so probe direction depends on regularization and is not unique.', cross_layer='No summation or concatenation.', auc='Pooled OOF margins from different fitted folds; fold AUC also provided; raw edge dot products have no calibrated threshold.')
    (out / 'analysis.json').write_text(json.dumps(details, indent=2) + '\n')
    plot(summaries, directions, out)


def plot(summary, directions, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for readout in ['speaker', 'statement', 'joint']:
        for ax, metric in zip(axes[:2], ['accuracy', 'roc_auc']):
            rs = [r for r in summary if r['readout'] == readout and r['metric'] == metric]
            ax.plot([r['layer'] for r in rs], [r['mean'] for r in rs], 'o-', label=readout)
            ax.fill_between([r['layer'] for r in rs], [r['ci_low'] for r in rs], [r['ci_high'] for r in rs], alpha=.1)
    for axis in ['lm_raw', 'lm_norm_weighted']:
        rs = [r for r in summary if r['readout'] == axis and r['metric'] == 'roc_auc']
        if rs:
            axes[1].plot([r['layer'] for r in rs], [r['mean'] for r in rs], '--', label=axis)
        layers = sorted({r['layer'] for r in directions if r['axis'] == axis})
        if layers:
            means = [np.mean([r['cosine'] for r in directions if r['axis'] == axis and r['layer'] == layer and r['regime'] == 'joint']) for layer in layers]
            axes[2].plot(layers, means, 'o-', label=axis)
    for ax, title in zip(axes, ['Held-out accuracy', 'Pooled OOF / LM-axis ROC-AUC', 'Joint-fold mean direction cosine']):
        ax.set(title=title, xlabel='Decoder block (post-o_proj edge update)')
        ax.axhline(0 if ax is axes[2] else .5, color='gray', ls=':')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / 'edge_probes.png', dpi=180)
    fig.savefig(out / 'edge_probes.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representations', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    analyze(args.representations, args.metadata, args.output_dir, args.bootstrap)
