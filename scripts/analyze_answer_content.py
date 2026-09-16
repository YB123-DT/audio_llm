#!/usr/bin/env python3
"""Matched within-content vs cross-content complete-answer linear readability."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

if __package__:
    from .analyze_module_b import validate
    from .analyze_edge_representation import auc, folds, write_csv
else:
    from analyze_module_b import validate
    from analyze_edge_representation import auc, folds, write_csv

SEED = 20260915


def within_folds(rows):
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    for fold, cross_train, test in folds(rows, 'joint'):
        actor, statement = actors[test[0]], statements[test[0]]
        train = np.flatnonzero((actors != actor) & (statements == statement))
        if len(train) != len(cross_train):
            raise ValueError('Within/cross training sample counts differ')
        yield fold, train, test


def read_cross(path, rows, keys):
    """Reject missing/duplicate/extra records and any mismatched test identity."""
    source = list(csv.DictReader(Path(path).open()))
    expected = {(position, layer, r['sample_id']) for position, layer in keys for r in rows}
    lookup = {r['sample_id']: r for r in rows}
    result = {}
    for r in source:
        if r['regime'] != 'joint' or r['position'] not in ('answer', 'answer_final_norm'):
            continue
        key = (r['position'], int(r['layer']), r['sample_id'])
        if key not in expected or key in result:
            raise ValueError('Unexpected or duplicate cross prediction')
        meta = lookup[r['sample_id']]
        if any(r[k] != meta[k] for k in ('actor', 'statement', 'emotion')):
            raise ValueError('Cross prediction metadata mismatch')
        if r['fold'] != f"actor_{meta['actor']}_statement_{meta['statement']}":
            raise ValueError('Cross test fold mismatch')
        score = float(r['score'])
        if not np.isfinite(score) or r['prediction'] != ('happy' if score > 0 else 'sad'):
            raise ValueError('Invalid cross score/prediction')
        result[key] = score
    if set(result) != expected:
        raise ValueError('Incomplete cross prediction coverage')
    return result


def metric_draws(y, scores, actors, statements, counts):
    """Identical actor draws retain both statements; folds are actor x statement."""
    unique = sorted(set(actors))
    indices = [np.flatnonzero(actors == actor) for actor in unique]
    sizes = np.array([len(i) for i in indices])
    correct = np.array([((scores[i] > 0) == y[i]).sum() for i in indices])
    hs = [scores[i][y[i] == 1] for i in indices]
    ss = [scores[i][y[i] == 0] for i in indices]
    kernel = np.array([[((h[:, None] > s).astype(float) + .5 * (h[:, None] == s)).sum() for s in ss] for h in hs])
    pooled = np.einsum('bi,ij,bj->b', counts, kernel, counts) / ((counts @ np.array([len(h) for h in hs])) * (counts @ np.array([len(s) for s in ss])))
    actor_auc = np.array([np.mean([auc(y[j], scores[j]) for statement in sorted(set(statements[i])) for j in [i[statements[i] == statement]]]) for i in indices])
    return {
        'mean_within_fold_auc': (float(actor_auc.mean()), counts @ actor_auc / counts.sum(axis=1)),
        'pooled_roc_auc': (auc(y, scores), pooled),
        'accuracy': (float(((scores > 0) == y).mean()), counts @ correct / (counts @ sizes)),
    }


def paired_summary(y, within, cross, actors, statements, counts):
    wm = metric_draws(y, within, actors, statements, counts)
    cm = metric_draws(y, cross, actors, statements, counts)
    result = []
    for metric in wm:
        for condition, mean, draws in [('within', *wm[metric]), ('cross', *cm[metric]), ('gap', wm[metric][0] - cm[metric][0], wm[metric][1] - cm[metric][1])]:
            low, high = np.quantile(draws, [.025, .975])
            result.append(dict(metric=metric, condition=condition, mean=mean, ci_low=float(low), ci_high=float(high)))
    return result


def plot_summary(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey='row')
    colors = {'within': '#2675ac', 'cross': '#cb6434', 'gap': '#68449b'}
    for col, statement in enumerate(('01', '02', 'combined')):
        for row, conditions in enumerate((('within', 'cross'), ('gap',))):
            ax = axes[row, col]
            for condition in conditions:
                points = [r for r in rows if r['statement'] == statement and r['metric'] == 'mean_within_fold_auc' and r['condition'] == condition]
                blocks = [r for r in points if r['position'] == 'answer']
                x = [r['layer'] for r in blocks]
                ax.plot(x, [r['mean'] for r in blocks], color=colors[condition], label=condition.capitalize())
                ax.fill_between(x, [r['ci_low'] for r in blocks], [r['ci_high'] for r in blocks], color=colors[condition], alpha=.14)
                norm = next(r for r in points if r['position'] == 'answer_final_norm')
                ax.errorbar([25], [norm['mean']], yerr=[[norm['mean'] - norm['ci_low']], [norm['ci_high'] - norm['mean']]], fmt='D', color=colors[condition], markersize=4)
            ax.axhline(.5 if row == 0 else 0, color='gray', linestyle=':')
            ax.grid(alpha=.15)
            ax.set_xticks([0, 4, 8, 12, 16, 20, 23, 25], ['0', '4', '8', '12', '16', '20', '23', 'N'])
            if row == 0:
                ax.set_title('Statement ' + statement if statement != 'combined' else 'Both statements (macro)')
                ax.set_ylim(min(.35, min(r['ci_low'] for r in rows if r['metric'] == 'mean_within_fold_auc' and r['condition'] in ('within', 'cross')) - .02), 1.02)
            else:
                ax.set_xlabel('Decoder block (N: final RMSNorm)')
            if col == 0:
                ax.set_ylabel('Mean actor-fold ROC-AUC' if row == 0 else 'Within minus cross AUC')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Complete answer state: within-content vs cross-content (actor held out in both)')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def analyze(source, metadata, cross_path, cross_summary, out, bootstrap=10000):
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if bootstrap <= 0:
        raise ValueError('Positive bootstrap count required')
    data = np.load(source, allow_pickle=False)
    rows = list(csv.DictReader(metadata.open()))
    parity = validate(data, rows)
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    if len(rows) != 192 or len(set(actors)) != 24 or set(statements) != {'01', '02'}:
        raise ValueError('Expected 192 samples / 24 actors / two statements')
    readouts = [('answer', i, data['answer_states'][:, i]) for i in range(24)] + [('answer_final_norm', 24, data['answer_final_norm'])]
    cross = read_cross(cross_path, rows, [(p, l) for p, l, _ in readouts])
    reference = {(r['position'], int(r['layer']), r['metric']): float(r['mean']) for r in csv.DictReader(cross_summary.open()) if r['regime'] == 'joint'}
    rng = np.random.default_rng(SEED)
    counts = np.array([np.bincount(d, minlength=24) for d in rng.integers(0, 24, (bootstrap, 24))])
    splits = list(within_folds(rows))
    membership, predictions, fold_metrics, summary = [], [], [], []
    cross_splits = {name: train for name, train, _ in folds(rows, 'joint')}
    for fold, train, test in splits:
        if len(train) != 92 or len(test) != 4 or set(y[test]) != {0, 1}:
            raise ValueError('Unexpected fold counts/classes')
        for condition, train_indices in [('within', train), ('cross', cross_splits[fold])]:
            for subset, indices in [('train', train_indices), ('test', test)]:
                membership.extend(dict(condition=condition, fold=fold, subset=subset, sample_id=rows[i]['sample_id']) for i in indices)
    max_reference_difference = 0.
    for position, layer, vectors in readouts:
        x = vectors.astype(np.float64)
        within = np.full(len(rows), np.nan)
        cached = np.array([cross[position, layer, r['sample_id']] for r in rows])
        for fold, train, test in splits:
            scaler = StandardScaler().fit(x[train])
            clf = LogisticRegression(C=1., max_iter=5000, solver='lbfgs', random_state=SEED)
            clf.fit(scaler.transform(x[train]), y[train])
            if clf.n_iter_.max() >= 5000:
                raise RuntimeError('Probe failed to converge')
            within[test] = clf.decision_function(scaler.transform(x[test]))
            for condition, values in [('within', within[test]), ('cross', cached[test])]:
                fold_metrics.append(dict(position=position, layer=layer, condition=condition, fold=fold, actor=rows[test[0]]['actor'], statement=rows[test[0]]['statement'], n_train=len(train), n_test=len(test), accuracy=float(((values > 0) == y[test]).mean()), roc_auc=auc(y[test], values)))
                predictions.extend(dict(position=position, layer=layer, condition=condition, fold=fold, sample_id=rows[i]['sample_id'], actor=rows[i]['actor'], statement=rows[i]['statement'], emotion=rows[i]['emotion'], score=float(value), prediction='happy' if value > 0 else 'sad') for i, value in zip(test, values))
        if not np.isfinite(within).all():
            raise ValueError('Incomplete within predictions')
        for statement in ('01', '02', 'combined'):
            ix = np.arange(len(rows)) if statement == 'combined' else np.flatnonzero(statements == statement)
            values = paired_summary(y[ix], within[ix], cached[ix], actors[ix], statements[ix], counts)
            summary.extend(dict(position=position, layer=layer, statement=statement, **r) for r in values)
            if statement == 'combined':
                for r in values:
                    if r['condition'] == 'cross':
                        metric = 'roc_auc' if r['metric'] == 'pooled_roc_auc' else r['metric']
                        diff = abs(r['mean'] - reference[position, layer, metric])
                        max_reference_difference = max(max_reference_difference, diff)
                        if diff > 1e-12:
                            raise ValueError('Cached cross metrics do not reproduce Module B')
        print(f'Completed {position} {layer}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [('summary', summary), ('oof_predictions', predictions), ('fold_metrics', fold_metrics), ('fold_membership', membership)]:
        write_csv(out / (name + '.csv'), values)
    provenance = dict(n_samples=len(rows), n_actors=24, n_readouts=25, n_new_probe_fits=25 * 48, seed=SEED, bootstrap=bootstrap, sklearn_version=sklearn.__version__, answer_head_parity_max=parity, cross_module_b_metric_difference_max=max_reference_difference,
        sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, metadata, cross_path, cross_summary, Path(__file__))},
        protocol='Complete post-block answer states 0..23 and separate final RMSNorm. Within: train 23 other actors, same statement; cross: reuse Module B joint train 23 other actors, other statement. Both 92 train / identical 4 test. Train-only StandardScaler; C=1 lbfgs max_iter=5000; happy=1; zero predicts sad.',
        primary_metric='Mean actor x statement fold AUC. Combined is macro mean of two statements, each mean of 24 actor-fold AUCs. Pooled OOF AUC separately reported and can reflect offsets between fitted models.',
        uncertainty='Paired actor-cluster percentile 95% bootstrap, 10000 draws by default, both statements retained together, same draws in within and cross then subtract; fixed fitted probes, no refits, pointwise unadjusted.',
        scope='Observational linear readability, not causal explanation or proof of direction rotation. No model forward or tuning.')
    (out / 'analysis.json').write_text(json.dumps(provenance, indent=2) + '\n')
    plot_summary(summary, out / 'within_cross_content.png')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('representations', 'metadata', 'cross-predictions', 'cross-summary', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    analyze(args.representations, args.metadata, args.cross_predictions, args.cross_summary, args.output_dir, args.bootstrap)
