#!/usr/bin/env python3
"""Fixed early answer-update intervention: within-content actor-held-out probes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_answer_content import SEED, within_folds, metric_draws
    from .analyze_edge_representation import auc, write_csv
else:
    from analyze_answer_content import SEED, within_folds, metric_draws
    from analyze_edge_representation import auc, write_csv

CONDITIONS = ('clean', 'no_attn', 'no_mlp')
ENDPOINTS = {'block6': ('answer', 6), 'final_answer': ('answer_final_norm', 24)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_states(data, clean, rows):
    ids = np.array([r['sample_id'] for r in rows])
    if len(set(ids)) != len(ids) or not np.array_equal(data['sample_ids'], ids) or not np.array_equal(clean['sample_ids'], ids):
        raise ValueError('Sample identity/order mismatch')
    if tuple(data['conditions']) != CONDITIONS:
        raise ValueError('Unexpected intervention conditions/order')
    references = {'block6': clean['answer_states'][:, 6], 'final_answer': clean['answer_final_norm']}
    for endpoint, reference in references.items():
        vectors = data[endpoint]
        if vectors.shape != (len(rows), 3, reference.shape[1]) or not np.isfinite(vectors).all():
            raise ValueError('Invalid intervention vectors')
        if not np.array_equal(vectors[:, 0], reference):
            raise ValueError('Clean vectors differ from original probe cache: ' + endpoint)


def validate_cache_provenance(analysis_path, source, metadata):
    info = json.loads(Path(analysis_path).read_text())
    hashes = info['sources_sha256']
    for path in (source, metadata, Path(__file__).with_name('analyze_answer_content.py')):
        matches = [v for k, v in hashes.items() if Path(k).name == Path(path).name]
        if len(matches) != 1 or matches[0] != sha(path):
            raise ValueError('Cached probe provenance hash mismatch: ' + str(path))
    if info['seed'] != SEED or info['n_samples'] != 192 or info['n_actors'] != 24:
        raise ValueError('Cached probe protocol mismatch')
    return info


def read_clean_predictions(path, rows):
    expected = {(p, l, r['sample_id']) for p, l in ENDPOINTS.values() for r in rows}
    lookup = {r['sample_id']: r for r in rows}
    result = {}
    for r in csv.DictReader(Path(path).open()):
        if r['condition'] != 'within' or (r['position'], int(r['layer'])) not in ENDPOINTS.values():
            continue
        key = (r['position'], int(r['layer']), r['sample_id'])
        if key not in expected or key in result:
            raise ValueError('Unexpected/duplicate clean prediction')
        meta = lookup[r['sample_id']]
        if any(r[k] != meta[k] for k in ('actor', 'statement', 'emotion')) or r['fold'] != f"actor_{meta['actor']}_statement_{meta['statement']}":
            raise ValueError('Clean prediction identity/fold mismatch')
        score = float(r['score'])
        if not np.isfinite(score) or r['prediction'] != ('happy' if score > 0 else 'sad'):
            raise ValueError('Invalid cached score')
        result[key] = score
    if set(result) != expected:
        raise ValueError('Incomplete cached prediction coverage')
    return {name: np.array([result[p, l, r['sample_id']] for r in rows]) for name, (p, l) in ENDPOINTS.items()}


def fit_probe(x, y, train, test):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if np.intersect1d(train, test).size:
        raise ValueError('Train/test overlap')
    scaler = StandardScaler().fit(x[train])
    clf = LogisticRegression(C=1., solver='lbfgs', max_iter=5000, random_state=SEED)
    clf.fit(scaler.transform(x[train]), y[train])
    if clf.n_iter_.max() >= 5000:
        raise RuntimeError('Probe did not converge')
    return clf.decision_function(scaler.transform(x[test]))


def summarize(y, scores, actors, statements, counts):
    metrics = {condition: metric_draws(y, scores[condition], actors, statements, counts) for condition in CONDITIONS}
    summary, contrasts = [], []
    for condition, values in metrics.items():
        for metric, (mean, draws) in values.items():
            low, high = np.quantile(draws, [.025, .975])
            summary.append(dict(condition=condition, metric=metric, mean=mean, ci_low=float(low), ci_high=float(high)))
    for label, left, right in [('delta_A', 'no_attn', 'clean'), ('delta_A_minus_M', 'no_attn', 'no_mlp'), ('delta_M_secondary', 'no_mlp', 'clean')]:
        for metric in metrics[left]:
            lm, ld = metrics[left][metric]
            rm, rd = metrics[right][metric]
            low, high = np.quantile(ld - rd, [.025, .975])
            contrasts.append(dict(contrast=label, left=left, right=right, metric=metric, mean=lm-rm, ci_low=float(low), ci_high=float(high)))
    return summary, contrasts


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    lower = min(.35, min(r['ci_low'] for r in summary if r['metric'] == 'mean_within_fold_auc') - .03)
    for row, endpoint in enumerate(ENDPOINTS):
        for col, statement in enumerate(('01', '02', 'combined')):
            ax = axes[row, col]
            points = [next(r for r in summary if r['endpoint'] == endpoint and r['statement'] == statement and r['metric'] == 'mean_within_fold_auc' and r['condition'] == c) for c in CONDITIONS]
            means = np.array([r['mean'] for r in points])
            ax.errorbar(range(3), means, yerr=[means - [r['ci_low'] for r in points], np.array([r['ci_high'] for r in points]) - means], fmt='o', capsize=4, color='#2675ac')
            ax.set_xticks(range(3), ['Clean', 'No-attn', 'No-MLP'])
            ax.set_ylim(lower, 1.02)
            ax.axhline(.5, color='gray', ls=':')
            ax.grid(alpha=.15)
            ax.set_title(('Statement ' + statement) if statement != 'combined' else 'Both statements (macro)')
            if col == 0:
                ax.set_ylabel(('Primary: block 6' if row == 0 else 'Secondary: final RMSNorm') + '\nMean actor-fold AUC')
    fig.suptitle('Answer updates blocked only at blocks 1–6; external within-content probes')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def analyze(args):
    import sklearn
    data = np.load(args.representations, allow_pickle=False)
    clean = np.load(args.clean_representations, allow_pickle=False)
    rows = list(csv.DictReader(args.metadata.open()))
    validate_states(data, clean, rows)
    info = validate_cache_provenance(args.clean_analysis, args.clean_representations, args.metadata)
    if info['sklearn_version'] != sklearn.__version__:
        raise ValueError('Cached/new probes require identical sklearn version')
    cached = read_clean_predictions(args.clean_predictions, rows)
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    if len(rows) != 192 or len(set(actors)) != 24 or set(statements) != {'01', '02'} or args.bootstrap <= 0:
        raise ValueError('Expected full 192-sample protocol and positive bootstrap')
    splits = list(within_folds(rows))
    membership, predictions, fold_metrics, summary, contrasts = [], [], [], [], []
    for fold, train, test in splits:
        if len(train) != 92 or len(test) != 4 or set(y[test]) != {0, 1}:
            raise ValueError('Unexpected fold size/classes')
        membership.extend(dict(fold=fold, subset=subset, sample_id=rows[i]['sample_id']) for subset, indices in [('train', train), ('test', test)] for i in indices)
    rng = np.random.default_rng(SEED)
    counts = np.array([np.bincount(d, minlength=24) for d in rng.integers(0, 24, (args.bootstrap, 24))])
    for endpoint in ENDPOINTS:
        scores = {c: np.full(len(rows), np.nan) for c in CONDITIONS}
        scores['clean'] = cached[endpoint]
        for cidx, condition in enumerate(CONDITIONS):
            x = data[endpoint][:, cidx].astype(np.float64)
            for fold, train, test in splits:
                if condition != 'clean':
                    scores[condition][test] = fit_probe(x, y, train, test)
                values = scores[condition][test]
                fold_metrics.append(dict(endpoint=endpoint, condition=condition, fold=fold, actor=rows[test[0]]['actor'], statement=rows[test[0]]['statement'], n_train=len(train), n_test=len(test), accuracy=float(((values > 0) == y[test]).mean()), roc_auc=auc(y[test], values)))
                predictions.extend(dict(endpoint=endpoint, condition=condition, fold=fold, **{k: rows[i][k] for k in ('sample_id', 'actor', 'statement', 'emotion')}, score=float(value), prediction='happy' if value > 0 else 'sad') for i, value in zip(test, values))
            if not np.isfinite(scores[condition]).all():
                raise ValueError('Incomplete OOF scores')
        for statement in ('01', '02', 'combined'):
            ix = np.arange(len(rows)) if statement == 'combined' else np.flatnonzero(statements == statement)
            sm, ct = summarize(y[ix], {c: s[ix] for c, s in scores.items()}, actors[ix], statements[ix], counts)
            summary.extend(dict(endpoint=endpoint, statement=statement, **r) for r in sm)
            contrasts.extend(dict(endpoint=endpoint, statement=statement, **r) for r in ct)
        print('Completed ' + endpoint, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [('summary', summary), ('contrasts', contrasts), ('oof_predictions', predictions), ('fold_metrics', fold_metrics), ('fold_membership', membership)]:
        write_csv(args.output_dir / (name + '.csv'), values)
    provenance = dict(n_samples=192, n_actors=24, n_new_probe_fits=192, clean_probe_fits_reused=96, seed=SEED, bootstrap=args.bootstrap, sklearn_version=sklearn.__version__,
        clean_vectors_exact_match=True, sources_sha256={str(p): sha(p) for p in (args.representations, args.clean_representations, args.metadata, args.clean_predictions, args.clean_analysis, Path(__file__), Path(__file__).with_name('analyze_answer_content.py'))},
        protocol='Within each statement, leave one actor out: 92 train/4 test. Train-only StandardScaler; LogisticRegression C=1 lbfgs max_iter=5000; happy=1. Separate external classifier fits for each intervention and endpoint. Clean cached predictions reused only after exact vector/identity and provenance parity.',
        primary='Block 6 mean actor x statement fold AUC; contrasts no_attn-clean and no_attn-no_mlp. Combined macro mean; statement-specific breakdown.',
        secondary='Final RMSNorm answer state. No native LM-head performance claim. Pooled AUC/accuracy and no_mlp-clean are descriptive secondary metrics.',
        uncertainty='Paired actor-cluster percentile 95% bootstrap, same 10000 draws by default across conditions/endpoints/statements; both statements of each actor retained; fixed fitted probes, no bootstrap refits, unadjusted intervals.',
        scope='Three fixed conditions only; external linear readability under intervention, not semantic source attribution or proof of information deletion.')
    (args.output_dir / 'analysis.json').write_text(json.dumps(provenance, indent=2) + '\n')
    plot(summary, args.output_dir / 'early_answer_causal.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('representations', 'metadata', 'clean-representations', 'clean-predictions', 'clean-analysis', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    analyze(parser.parse_args())
