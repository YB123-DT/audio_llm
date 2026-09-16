#!/usr/bin/env python3
"""Module A protocol applied to complete decoder audio and answer states."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_module_a_baseline import validate as validate_endpoint
    from .analyze_edge_representation import folds, auc, clustered_metrics, write_csv
else:
    from analyze_module_a_baseline import validate as validate_endpoint
    from analyze_edge_representation import folds, auc, clustered_metrics, write_csv

REGIMES = ('speaker', 'statement', 'joint')


def validate(data, rows):
    endpoint = dict(sample_ids=data['sample_ids'], projector=data['projector'],
                    answer_state=data['answer_final_norm'], d_lm=data['d_lm'],
                    clean_margins=data['clean_margins'])
    parity = validate_endpoint(endpoint, rows)
    shape = (len(rows), 24, len(data['d_lm']))
    for name in ('audio_states', 'answer_states'):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError('Invalid 24-layer representation: ' + name)
    return parity


def representations(data):
    yield 'projector', -1, data['projector']
    for position, key in [('audio', 'audio_states'), ('answer', 'answer_states')]:
        for layer in range(24):
            yield position, layer, data[key][:, layer, :]
    yield 'answer_final_norm', 24, data['answer_final_norm']


def compare_endpoints(summary, module_a_summary, tolerance=1e-10):
    """Require exact projector metrics; disclose normalized-answer numerical drift."""
    reference = list(csv.DictReader(Path(module_a_summary).open()))
    lookup = {(r['object'], r['regime'], r['metric']): r for r in reference}
    differences = []
    answer_differences = []
    mapping = {'projector': 'projector', 'answer_final_norm': 'answer_state'}
    for row in summary:
        if row['position'] in mapping:
            key = (mapping[row['position']], row['regime'], row['metric'])
            for metric_field in ('mean', 'ci_low', 'ci_high'):
                if row[metric_field] == '':
                    continue
                difference = abs(float(row[metric_field]) - float(lookup[key][metric_field]))
                differences.append(difference)
                if not np.isfinite(difference):
                    raise ValueError(f'Nonfinite Module A endpoint difference {key}')
                if row['position'] == 'projector' and difference > tolerance:
                    raise ValueError(f'Module A endpoint mismatch {key} {metric_field}: {difference}')
                if row['position'] == 'answer_final_norm':
                    answer_differences.append(dict(regime=row['regime'], metric=row['metric'], field=metric_field,
                                                   module_a=float(lookup[key][metric_field]), module_b=float(row[metric_field]),
                                                   difference=float(row[metric_field]) - float(lookup[key][metric_field])))
    if len(differences) not in (18, 42):
        raise ValueError('Incomplete Module A endpoint comparison')
    return dict(n_comparisons=len(differences), max_absolute_difference=max(differences), projector_tolerance=tolerance,
                answer_exact_match=all(r['difference'] == 0 for r in answer_differences),
                answer_differences=answer_differences,
                answer_policy='Fresh actual final RMSNorm output retained; no posthoc metric threshold. Inspect extraction all-sample vector/head parity and these disclosed differences.')


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey='row')
    colors = {'audio': '#2675ac', 'answer': '#cb6434'}
    for col, regime in enumerate(REGIMES):
        for row_index, metric in enumerate(('accuracy', 'roc_auc')):
            ax = axes[row_index, col]
            selected = [r for r in summary if r['regime'] == regime and r['metric'] == metric]
            for position in ('audio', 'answer'):
                points = sorted([r for r in selected if r['position'] == position], key=lambda r: r['layer'])
                if position == 'audio':
                    points = [r for r in selected if r['position'] == 'projector'] + points
                x = np.array([r['layer'] for r in points])
                y = np.array([r['mean'] for r in points])
                ax.plot(x, y, color=colors[position], marker='.', markersize=4,
                        label='Audio-token mean (+ projector)' if position == 'audio' else 'Complete answer state (post-block)')
                if all(r['ci_low'] != '' for r in points):
                    ax.fill_between(x, [r['ci_low'] for r in points], [r['ci_high'] for r in points], color=colors[position], alpha=.12)
            if metric == 'roc_auc':
                for position in ('audio', 'answer'):
                    fold_points = sorted([r for r in summary if r['regime'] == regime and r['metric'] == 'mean_within_fold_auc' and r['position'] == position], key=lambda r: r['layer'])
                    if position == 'audio':
                        fold_points = [r for r in summary if r['regime'] == regime and r['metric'] == 'mean_within_fold_auc' and r['position'] == 'projector'] + fold_points
                    ax.plot([r['layer'] for r in fold_points], [r['mean'] for r in fold_points], color=colors[position], linestyle='--', linewidth=1.4)
                norm_fold = next(r for r in summary if r['regime'] == regime and r['metric'] == 'mean_within_fold_auc' and r['position'] == 'answer_final_norm')
                ax.scatter([26], [norm_fold['mean']], facecolors='none', edgecolors=colors['answer'], marker='D', s=28)
            endpoint = next(r for r in selected if r['position'] == 'answer_final_norm')
            ax.scatter([26], [endpoint['mean']], color=colors['answer'], marker='D', s=28, label='Answer after final RMSNorm')
            if endpoint['ci_low'] != '':
                ax.vlines(26, endpoint['ci_low'], endpoint['ci_high'], colors=colors['answer'], alpha=.6)
            projector = next(r for r in selected if r['position'] == 'projector')
            ax.scatter([-1], [projector['mean']], color=colors['audio'], marker='s', s=28)
            ax.axhline(.5, color='gray', linestyle=':', linewidth=1)
            metric_values = [r['mean'] for r in summary if r['metric'] == metric]
            metric_values += [r['ci_low'] for r in summary if r['metric'] == metric and r['ci_low'] != '']
            ax.set_ylim(min(.3, min(metric_values) - .03), 1.02)
            ax.grid(alpha=.15)
            ax.set_xticks([-1, 0, 4, 8, 12, 16, 20, 23, 26], ['P', '0', '4', '8', '12', '16', '20', '23', 'N'])
            if row_index == 0:
                ax.set_title(regime.capitalize() + '-held-out')
            else:
                ax.set_xlabel('Decoder block (P: projector; N: final norm)')
            if col == 0:
                ax.set_ylabel('Accuracy' if metric == 'accuracy' else 'ROC-AUC (solid: pooled OOF)' )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', linestyle='--'))
    labels.append('Dashed/open: mean within-fold AUC')
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Module B: matched-protocol emotion readability through the decoder')
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(output, dpi=180)
    fig.savefig(Path(output).with_suffix('.pdf'))
    plt.close(fig)


def analyze(source, metadata, out, module_a_summary, bootstrap=10000, make_plot=True):
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    data = np.load(source, allow_pickle=False)
    rows = list(csv.DictReader(Path(metadata).open()))
    parity = validate(data, rows)
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    actors = np.array([r['actor'] for r in rows])
    unique = sorted(set(actors))
    draws = np.random.default_rng(20260915).integers(0, len(unique), (bootstrap, len(unique)))
    counts = np.array([np.bincount(d, minlength=len(unique)) for d in draws])
    summaries, predictions, fold_metrics, split_rows = [], [], [], []
    splits = {regime: list(folds(rows, regime)) for regime in REGIMES}
    for regime, split in splits.items():
        for fold, train, test in split:
            for subset, indices in [('train', train), ('test', test)]:
                split_rows.extend(dict(regime=regime, fold=fold, subset=subset, sample_id=rows[i]['sample_id']) for i in indices)
    for position, layer, vectors in representations(data):
        x = vectors.astype(np.float64)
        for regime, split in splits.items():
            scores = np.full(len(y), np.nan)
            coverage = np.zeros(len(y), dtype=int)
            fold_aucs = []
            for fold, train, test in split:
                scaler = StandardScaler().fit(x[train])
                clf = LogisticRegression(C=1., max_iter=5000, solver='lbfgs', random_state=20260915)
                clf.fit(scaler.transform(x[train]), y[train])
                if clf.n_iter_.max() >= 5000:
                    raise RuntimeError('Probe did not converge')
                values = clf.decision_function(scaler.transform(x[test]))
                scores[test] = values
                coverage[test] += 1
                fold_auc = auc(y[test], values)
                fold_aucs.append(fold_auc)
                fold_metrics.append(dict(position=position, layer=layer, regime=regime, fold=fold,
                                         n_train=len(train), n_test=len(test), accuracy=float(((values > 0) == y[test]).mean()), roc_auc=fold_auc))
                predictions.extend(dict(position=position, layer=layer, regime=regime, fold=fold,
                                        sample_id=rows[i]['sample_id'], actor=rows[i]['actor'], statement=rows[i]['statement'],
                                        emotion=rows[i]['emotion'], score=float(value), prediction='happy' if value > 0 else 'sad')
                                   for i, value in zip(test, values))
            if not np.all(coverage == 1) or not np.isfinite(scores).all():
                raise ValueError('OOF coverage failed')
            metrics = clustered_metrics(y, scores, actors, counts) if bootstrap else [
                dict(metric='accuracy', mean=float(((scores > 0) == y).mean()), ci_low='', ci_high=''),
                dict(metric='roc_auc', mean=auc(y, scores), ci_low='', ci_high='')]
            summaries.extend(dict(position=position, layer=layer, regime=regime, **m) for m in metrics)
            summaries.append(dict(position=position, layer=layer, regime=regime, metric='mean_within_fold_auc',
                                  mean=float(np.mean(fold_aucs)), ci_low='', ci_high=''))
            print(f'Completed {position} layer {layer} {regime}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in [('summary', summaries), ('oof_predictions', predictions), ('fold_metrics', fold_metrics), ('fold_membership', split_rows)]:
        write_csv(out / (filename + '.csv'), values)
    comparison = compare_endpoints(summaries, module_a_summary)
    provenance = dict(n_samples=len(rows), n_actors=len(unique), n_representation_readouts=50,
                      answer_head_parity_max=parity, module_a_endpoint_comparison=comparison,
                      sources_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (source, metadata, module_a_summary, Path(__file__))},
                      sklearn_version=sklearn.__version__, bootstrap=bootstrap, seed=20260915,
                      protocol='Exact Module A folds and metrics; train-only StandardScaler; LogisticRegression C=1 lbfgs max_iter=5000; happy=1; zero predicts sad; no tuning.',
                      positions='Audio-token mean and complete answer position, after each decoder block 0..23 before final RMSNorm. Final RMSNorm answer shown separately, not as a 25th decoder block.',
                      uncertainty='Actor-cluster percentile 95%, fixed fitted folds, no refitting, pointwise unadjusted; within-fold AUC descriptive.',
                      pooled_auc='OOF scores come from different fitted models; pooled AUC can reflect fold offsets. Mean within-fold AUC also saved.',
                      scope='Observational linear readability only; not a causal localization or proof that information is destroyed.')
    (out / 'analysis.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if make_plot:
        plot_summary(summaries, out / 'decoder_readability.png')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representations', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--module-a-summary', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    analyze(args.representations, args.metadata, args.output_dir, args.module_a_summary, args.bootstrap)
