#!/usr/bin/env python3
"""Observational answer readability before/after attention and MLP; no ablation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

if __package__:
    from .analyze_answer_content import SEED, within_folds, metric_draws
    from .analyze_edge_representation import auc, folds, write_csv
    from .analyze_module_b import validate as validate_reference
else:
    from analyze_answer_content import SEED, within_folds, metric_draws
    from analyze_edge_representation import auc, folds, write_csv
    from analyze_module_b import validate as validate_reference


def validate_states(data, reference, rows):
    validate_reference(reference, rows)
    if list(data['sample_ids']) != [r['sample_id'] for r in rows]:
        raise ValueError('Sample identity/order mismatch')
    shape = reference['answer_states'].shape
    for name in ('answer_in', 'answer_attn', 'answer_out'):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError('Invalid state array ' + name)
    checks = [('post_mlp_reference', data['answer_out'], reference['answer_states']),
              ('residual_continuity', data['answer_in'][:, 1:], data['answer_out'][:, :-1]),
              ('final_norm', data['answer_final_norm'], reference['answer_final_norm']),
              ('margins', data['clean_margins'], reference['clean_margins']),
              ('head_direction', data['d_lm'], reference['d_lm'])]
    for name, value, expected in checks:
        if not np.array_equal(value, expected):
            raise ValueError('Exact vector parity failed: ' + name)
    return {name: 0. for name, _, _ in checks}


def read_cached(path, rows):
    lookup = {r['sample_id']: r for r in rows}
    expected = {(l, c, sid) for l in range(24) for c in ('within', 'cross') for sid in lookup}
    result = {}
    for r in csv.DictReader(Path(path).open()):
        if r['position'] == 'answer_final_norm':
            continue
        key = (int(r['layer']), r['condition'], r['sample_id'])
        if r['position'] != 'answer' or key not in expected or key in result:
            raise ValueError('Unexpected/duplicate cached prediction')
        meta = lookup[r['sample_id']]
        if any(r[k] != meta[k] for k in ('actor', 'statement', 'emotion')):
            raise ValueError('Cached metadata mismatch')
        if r['fold'] != f"actor_{meta['actor']}_statement_{meta['statement']}":
            raise ValueError('Cached fold mismatch')
        score = float(r['score'])
        if not np.isfinite(score) or r['prediction'] != ('happy' if score > 0 else 'sad'):
            raise ValueError('Invalid cached score')
        result[key] = score
    if set(result) != expected:
        raise ValueError('Incomplete cached predictions')
    return result


def contrast(after, before):
    """Subtract paired bootstrap draws, never confidence interval endpoints."""
    mean = after[0] - before[0]
    draws = after[1] - before[1]
    low, high = np.quantile(draws, [.025, .975])
    return dict(mean=float(mean), ci_low=float(low), ci_high=float(high))


def plot(summary, deltas, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True, sharey='row')
    colors = {'in': '#777777', 'attn': '#2675ac', 'out': '#cb6434'}
    for col, statement in enumerate(('01', '02', 'combined')):
        for row, condition in ((0, 'within'), (2, 'gap')):
            ax = axes[row, col]
            for state in ('in', 'attn', 'out'):
                points = [r for r in summary if r['statement'] == statement and r['state'] == state and r['condition'] == condition and r['metric'] == 'mean_within_fold_auc']
                ax.plot([r['layer'] for r in points], [r['mean'] for r in points], '.-', color=colors[state], label={'in': 'Block input', 'attn': 'Post-attention', 'out': 'Post-MLP'}[state], markersize=3)
            ax.axhline(.5 if row == 0 else 0, linestyle=':', color='gray')
        for row, condition in ((1, 'within'), (3, 'gap')):
            ax = axes[row, col]
            for component, color, shift in [('attention', colors['attn'], -.12), ('mlp', colors['out'], .12)]:
                points = [r for r in deltas if r['statement'] == statement and r['component'] == component and r['condition'] == condition and r['metric'] == 'mean_within_fold_auc']
                x = np.array([r['layer'] for r in points]) + shift
                means = np.array([r['mean'] for r in points])
                ax.errorbar(x, means, yerr=[means - np.array([r['ci_low'] for r in points]), np.array([r['ci_high'] for r in points]) - means], fmt='.', color=color, label=component, capsize=1, linewidth=.7)
            ax.axhline(0, linestyle=':', color='gray')
        axes[0, col].set_title('Statement ' + statement if statement != 'combined' else 'Both statements (macro)')
        axes[3, col].set_xlabel('Decoder block (0: first audio integration)')
        for ax in axes[:, col]:
            ax.grid(alpha=.15)
            ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    for row, label in enumerate(('Within-content AUC W', 'Change in W', 'Content gap G = W − C', 'Change in G')):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Normal forward: complete answer state at attention/MLP boundaries\nPaired actor-bootstrap 95% intervals for changes; pointwise, unadjusted')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def analyze(source, reference_path, metadata, cached_path, output, bootstrap=10000):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if bootstrap <= 0:
        raise ValueError('Positive bootstrap count required')
    data, reference = np.load(source, allow_pickle=False), np.load(reference_path, allow_pickle=False)
    rows = list(csv.DictReader(metadata.open()))
    parity = validate_states(data, reference, rows)
    cache_provenance_path = cached_path.parent / 'analysis.json'
    cache_provenance = json.loads(cache_provenance_path.read_text())
    known_hashes = set(cache_provenance['sources_sha256'].values())
    for required_source in (reference_path, metadata, Path(__file__).with_name('analyze_answer_content.py')):
        if hashlib.sha256(required_source.read_bytes()).hexdigest() not in known_hashes:
            raise ValueError('Cached analysis source/protocol hash mismatch: ' + str(required_source))
    if cache_provenance['seed'] != SEED:
        raise ValueError('Cached analysis seed differs')
    cached = read_cached(cached_path, rows)
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    if len(rows) != 192 or len(set(actors)) != 24 or set(statements) != {'01', '02'}:
        raise ValueError('Expected 192 samples / 24 actors / 2 statements')
    splits = {'within': list(within_folds(rows)), 'cross': list(folds(rows, 'joint'))}
    membership = []
    for condition, fs in splits.items():
        for fold, train, test in fs:
            if len(train) != 92 or len(test) != 4 or set(y[test]) != {0, 1}:
                raise ValueError('Unexpected fold sizes/classes')
            for subset, indices in [('train', train), ('test', test)]:
                membership.extend(dict(condition=condition, fold=fold, subset=subset, sample_id=rows[i]['sample_id']) for i in indices)
    rng = np.random.default_rng(SEED)
    counts = np.array([np.bincount(d, minlength=24) for d in rng.integers(0, 24, (bootstrap, 24))])
    summary, predictions, fold_metrics, metrics = [], [], [], {}
    n_fit = 0
    for layer in range(24):
        for state in ('in', 'attn', 'out'):
            x = data['answer_' + state][:, layer].astype(np.float64)
            scores = {}
            reused_layer = layer if state == 'out' else layer - 1 if state == 'in' and layer > 0 else None
            for condition, fs in splits.items():
                values = np.full(len(rows), np.nan)
                if reused_layer is not None:
                    values = np.array([cached[reused_layer, condition, r['sample_id']] for r in rows])
                for fold, train, test in fs:
                    if reused_layer is None:
                        scaler = StandardScaler().fit(x[train])
                        clf = LogisticRegression(C=1., max_iter=5000, solver='lbfgs', random_state=SEED)
                        clf.fit(scaler.transform(x[train]), y[train])
                        if clf.n_iter_.max() >= 5000:
                            raise RuntimeError('Probe failed to converge')
                        values[test] = clf.decision_function(scaler.transform(x[test]))
                        n_fit += 1
                    s = values[test]
                    fold_metrics.append(dict(layer=layer, state=state, condition=condition, fold=fold, actor=rows[test[0]]['actor'], statement=rows[test[0]]['statement'], n_train=92, n_test=4, accuracy=float(((s > 0) == y[test]).mean()), roc_auc=auc(y[test], s), source='new_fit' if reused_layer is None else f'cached_block_{reused_layer}'))
                    predictions.extend(dict(layer=layer, state=state, condition=condition, fold=fold, sample_id=rows[i]['sample_id'], actor=rows[i]['actor'], statement=rows[i]['statement'], emotion=rows[i]['emotion'], score=float(v), prediction='happy' if v > 0 else 'sad') for i, v in zip(test, s))
                if not np.isfinite(values).all():
                    raise ValueError('Incomplete predictions')
                scores[condition] = values
            for statement in ('01', '02', 'combined'):
                ix = np.arange(len(rows)) if statement == 'combined' else np.flatnonzero(statements == statement)
                result = {c: metric_draws(y[ix], scores[c][ix], actors[ix], statements[ix], counts) for c in scores}
                result['gap'] = {m: (result['within'][m][0] - result['cross'][m][0], result['within'][m][1] - result['cross'][m][1]) for m in result['within']}
                for condition, ms in result.items():
                    for metric, value in ms.items():
                        metrics[layer, state, statement, condition, metric] = value
                        low, high = np.quantile(value[1], [.025, .975])
                        summary.append(dict(layer=layer, state=state, statement=statement, condition=condition, metric=metric, mean=value[0], ci_low=float(low), ci_high=float(high)))
            print(f'Completed block {layer} {state}; fits {n_fit}', flush=True)
    deltas, aggregate = [], []
    telescoping_error = 0.
    for statement in ('01', '02', 'combined'):
        for condition in ('within', 'cross', 'gap'):
            for metric in ('mean_within_fold_auc', 'pooled_roc_auc', 'accuracy'):
                for component, before, after in [('attention', 'in', 'attn'), ('mlp', 'attn', 'out')]:
                    changes = []
                    for layer in range(24):
                        a, b = metrics[layer, after, statement, condition, metric], metrics[layer, before, statement, condition, metric]
                        deltas.append(dict(layer=layer, component=component, statement=statement, condition=condition, metric=metric, **contrast(a, b)))
                        changes.append((a[0] - b[0], a[1] - b[1]))
                    for start, end in [(0, 0), (1, 23)]:
                        values = changes[start:end + 1]
                        total = (sum(v[0] for v in values), sum((v[1] for v in values), np.zeros(bootstrap)))
                        aggregate.append(dict(blocks=f'{start}:{end}', component=component, statement=statement, condition=condition, metric=metric, **contrast(total, (0., np.zeros(bootstrap)))))
    for statement in ('01', '02', 'combined'):
        for condition in ('within', 'cross', 'gap'):
            for metric in ('mean_within_fold_auc', 'pooled_roc_auc', 'accuracy'):
                start = metrics[0, 'out', statement, condition, metric]
                end = metrics[23, 'out', statement, condition, metric]
                summed = np.zeros(bootstrap)
                point = 0.
                for layer in range(1, 24):
                    for before, after in [('in', 'attn'), ('attn', 'out')]:
                        a, b = metrics[layer, after, statement, condition, metric], metrics[layer, before, statement, condition, metric]
                        point += a[0] - b[0]
                        summed += a[1] - b[1]
                error = max(abs(point - (end[0] - start[0])), float(np.max(np.abs(summed - (end[1] - start[1])))))
                telescoping_error = max(telescoping_error, error)
                if error > 1e-12:
                    raise ValueError('Component sums failed paired telescoping check')
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('state_summary', summary), ('component_deltas', deltas), ('aggregate_deltas', aggregate), ('oof_predictions', predictions), ('fold_metrics', fold_metrics), ('fold_membership', membership)]:
        write_csv(output / (name + '.csv'), values)
    provenance = dict(n_samples=192, n_actors=24, n_states=72, new_probe_fits=n_fit, reused_state_count=47, seed=SEED, bootstrap=bootstrap, sklearn_version=sklearn.__version__, exact_vector_parity=parity, telescoping_max_error=telescoping_error,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, reference_path, metadata, cached_path, cache_provenance_path, Path(__file__))},
        protocol='Within: 23 other actors same statement; cross: 23 other actors other statement; both 92 train and identical 4 test. Train-only StandardScaler, C=1 logistic lbfgs max_iter=5000, happy=1. Zero score predicts sad.',
        primary='Mean actor x statement fold AUC. Combined macro-averages two statements. Gap within minus cross. Supplemental accuracy and pooled OOF AUC.',
        bootstrap_method='Shared actor-cluster draws across every state and condition, retaining both statements; fixed fitted probes, 95% percentile, pointwise unadjusted; deltas computed within each draw.',
        reuse='Exact Module B post-MLP vector and consecutive residual parity required before reusing within/cross OOF. Only 24 post-attention states and block 0 input fitted.',
        interpretation='Normal-forward observational probe differences, independently fitted readouts at each state, not causal ablation or additive mediation. Aggregate sums blocks 1..23 exclude block 0 initialization; final RMSNorm not part of sublayer contrasts.')
    (output / 'analysis.json').write_text(json.dumps(provenance, indent=2) + '\n')
    plot(summary, deltas, output / 'attention_mlp_readability.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('representations', 'reference-states', 'metadata', 'cached-predictions', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    analyze(args.representations, args.reference_states, args.metadata, args.cached_predictions, args.output_dir, args.bootstrap)
