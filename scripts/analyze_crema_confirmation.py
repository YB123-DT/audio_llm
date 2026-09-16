#!/usr/bin/env python3
"""Independent CREMA-D confirmation of the fixed early-answer intervention."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_early_answer_causal import CONDITIONS, SEED, fit_probe, sha, summarize
    from .analyze_edge_representation import auc, write_csv
else:
    from analyze_early_answer_causal import CONDITIONS, SEED, fit_probe, sha, summarize
    from analyze_edge_representation import auc, write_csv

ENDPOINTS = ('block6', 'final_answer')


def validate_manifest(rows):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty manifest or duplicate sample identity')
    actors = sorted({r['actor'] for r in rows})
    statements = sorted({r['statement'] for r in rows})
    if len(actors) < 2 or len(statements) < 2:
        raise ValueError('At least two actors and statements required')
    if len({r['intensity'] for r in rows}) != 1:
        raise ValueError('A single fixed intensity is required')
    keys = [(r['actor'], r['statement'], r['emotion']) for r in rows]
    expected = {(a, s, e) for a in actors for s in statements for e in ('happy', 'sad')}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Expected complete rectangular actor x statement x happy/sad coverage')
    return actors, statements


def within_folds(rows):
    """One matched pair per actor/statement; no actor or content leakage."""
    actors, statements = validate_manifest(rows)
    aa = np.array([r['actor'] for r in rows])
    ss = np.array([r['statement'] for r in rows])
    for actor in actors:
        for statement in statements:
            train = np.flatnonzero((aa != actor) & (ss == statement))
            test = np.flatnonzero((aa == actor) & (ss == statement))
            yield f'actor_{actor}_statement_{statement}', train, test


def validate_states(data, rows):
    validate_manifest(rows)
    if not np.array_equal(data['sample_ids'], [r['sample_id'] for r in rows]):
        raise ValueError('Sample identity/order mismatch')
    if tuple(data['conditions']) != CONDITIONS:
        raise ValueError('Unexpected condition order')
    dimension = None
    for endpoint in ENDPOINTS:
        x = data[endpoint]
        if x.ndim != 3 or x.shape[:2] != (len(rows), 3) or x.shape[2] < 1 or not np.isfinite(x).all():
            raise ValueError('Invalid endpoint vectors: ' + endpoint)
        if dimension is not None and x.shape[2] != dimension:
            raise ValueError('Endpoint dimension mismatch')
        dimension = x.shape[2]


def bootstrap_counts(n_actors, bootstrap):
    if bootstrap <= 0:
        raise ValueError('Positive bootstrap count required')
    draws = np.random.default_rng(SEED).integers(0, n_actors, (bootstrap, n_actors))
    return np.array([np.bincount(d, minlength=n_actors) for d in draws])


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, endpoint in zip(axes, ENDPOINTS):
        points = [next(r for r in summary if r['endpoint'] == endpoint and r['statement'] == 'combined' and r['metric'] == 'mean_within_fold_auc' and r['condition'] == c) for c in CONDITIONS]
        means = np.array([r['mean'] for r in points])
        ax.errorbar(range(3), means, yerr=[means - [r['ci_low'] for r in points], np.array([r['ci_high'] for r in points]) - means], fmt='o', capsize=4)
        for i, v in enumerate(means):
            ax.annotate(f'{v:.3f}', (i, v), xytext=(7, 0), textcoords='offset points', fontsize=9)
        ax.set_xticks(range(3), ['Clean', 'No-attn', 'No-MLP'])
        ax.set_xlim(-.35, 2.55)
        ax.set_ylim(0, 1.03)
        ax.axhline(.5, color='gray', ls=':')
        ax.grid(alpha=.15)
        ax.set_title('Primary: block 6' if endpoint == 'block6' else 'Secondary: final RMSNorm')
    axes[0].set_ylabel('Within-content mean actor-fold AUC')
    fig.suptitle('Independent CREMA-D confirmation: fixed blocks 1–6')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def analyze(args):
    import sklearn
    rows = list(csv.DictReader(args.metadata.open()))
    actor_names, statement_names = validate_manifest(rows)
    data = np.load(args.representations, allow_pickle=False)
    validate_states(data, rows)
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    splits = list(within_folds(rows))
    counts = bootstrap_counts(len(actor_names), args.bootstrap)
    membership, predictions, fold_metrics, summary, contrasts = [], [], [], [], []
    for fold, train, test in splits:
        membership.extend(dict(fold=fold, subset=subset, sample_id=rows[i]['sample_id']) for subset, ix in [('train', train), ('test', test)] for i in ix)
    for endpoint in ENDPOINTS:
        scores = {c: np.full(len(rows), np.nan) for c in CONDITIONS}
        for cidx, condition in enumerate(CONDITIONS):
            x = data[endpoint][:, cidx].astype(np.float64)
            for fold, train, test in splits:
                values = fit_probe(x, y, train, test)
                scores[condition][test] = values
                fold_metrics.append(dict(endpoint=endpoint, condition=condition, fold=fold, actor=rows[test[0]]['actor'], statement=rows[test[0]]['statement'], n_train=len(train), n_test=len(test), accuracy=float(((values > 0) == y[test]).mean()), roc_auc=auc(y[test], values)))
                predictions.extend(dict(endpoint=endpoint, condition=condition, fold=fold, **{k: rows[i][k] for k in ('sample_id', 'actor', 'statement', 'emotion')}, score=float(value), prediction='happy' if value > 0 else 'sad') for i, value in zip(test, values))
            if not np.isfinite(scores[condition]).all():
                raise ValueError('Incomplete OOF score coverage')
            print(f'Completed {endpoint} {condition}: {len(splits)} fits', flush=True)
        for statement in statement_names + ['combined']:
            ix = np.arange(len(rows)) if statement == 'combined' else np.flatnonzero(statements == statement)
            sm, ct = summarize(y[ix], {c: s[ix] for c, s in scores.items()}, actors[ix], statements[ix], counts)
            summary.extend(dict(endpoint=endpoint, statement=statement, **r) for r in sm)
            contrasts.extend(dict(endpoint=endpoint, statement=statement, **r) for r in ct)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [('summary', summary), ('contrasts', contrasts), ('oof_predictions', predictions), ('fold_metrics', fold_metrics), ('fold_membership', membership)]:
        write_csv(args.output_dir / (name + '.csv'), values)
    scripts = [Path(__file__).with_name(name) for name in ('analyze_early_answer_causal.py', 'analyze_answer_content.py', 'analyze_edge_representation.py')]
    info = dict(n_samples=len(rows), n_actors=len(actor_names), n_statements=len(statement_names), actors=actor_names, statements=statement_names, conditions=CONDITIONS, endpoints=ENDPOINTS, n_probe_fits=6 * len(splits), clean_probe_fits_reused=0, seed=SEED, bootstrap=args.bootstrap, sklearn_version=sklearn.__version__, numpy_version=np.__version__, sources_sha256={str(p): sha(p) for p in [args.representations, args.metadata, Path(__file__), *scripts]},
        protocol=f'Within-statement leave-one-actor-out: {2 * (len(actor_names)-1)} train, 2 test. Train-only StandardScaler, C=1 lbfgs max_iter=5000, happy=1; all conditions independently fitted with identical splits.',
        primary='Block6 mean actor x statement fold AUC; direct paired contrasts no_attn-clean and no_attn-no_mlp. Equal actor and statement weights; each two-sample fold AUC is 0, 0.5, or 1.',
        secondary='Final normalized answer-state readability; pooled OOF AUC, accuracy, and no_mlp-clean descriptive. No native LM-head performance claim.',
        uncertainty='Shared paired actor-cluster percentile 95% bootstrap across conditions/endpoints/statements; all statements retained per actor. Fixed fitted probes, no bootstrap refits; unadjusted intervals conditional on the fixed selected statements (statements are not resampled).',
        scope='Independent dataset confirmation for the same frozen checkpoint, fixed zero-based blocks1–6; no layer or condition reselection. Within-statement probes do not establish classifier transfer to unseen statements.')
    (args.output_dir / 'analysis.json').write_text(json.dumps(info, indent=2) + '\n')
    plot(summary, args.output_dir / 'crema_confirmation.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('representations', 'metadata', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    analyze(parser.parse_args())
