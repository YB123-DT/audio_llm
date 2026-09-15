#!/usr/bin/env python3
"""Evaluate frozen-model edge centering with paired actor-cluster uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
try:
    from scripts.analyze_clean_edge_gain import auc_samples, interval
    from scripts.analyze_residual_component_restore import write_csv
except ModuleNotFoundError:
    from analyze_clean_edge_gain import auc_samples, interval
    from analyze_residual_component_restore import write_csv

CONDITIONS = ['clean', 'zero_noop', 'center_joint', 'center_19', 'center_23']


def actor_metrics(values, baseline):
    gap = values[:, :, 0] - values[:, :, 1]
    shift = values - baseline
    return {
        'accuracy': np.stack([values[:, :, 0] > 0, values[:, :, 1] <= 0], axis=-1).mean(axis=(1, 2)),
        'happy_prediction_fraction': (values > 0).mean(axis=(1, 2)),
        'happy_margin': values[:, :, 0].mean(axis=1),
        'sad_margin': values[:, :, 1].mean(axis=1),
        'pair_separation': gap.mean(axis=1),
        'pair_direction_fraction': (gap > 0).mean(axis=1),
        'common_shift': shift.mean(axis=(1, 2)),
        'happy_shift': shift[:, :, 0].mean(axis=1),
        'sad_shift': shift[:, :, 1].mean(axis=1),
    }


def analyze(rows, out, reference):
    out = Path(out)
    actors = sorted({r['actor'] for r in rows})
    pairs = {a: sorted({r['pair_id'] for r in rows if r['actor'] == a}) for a in actors}
    if not actors or len({len(p) for p in pairs.values()}) != 1:
        raise ValueError('Balanced actor pairs required')
    lookup = {(r['condition'], r['actor'], r['pair_id'], r['emotion']): r for r in rows}
    if len(lookup) != len(rows) or {r['condition'] for r in rows} != set(CONDITIONS):
        raise ValueError('Duplicate rows or incorrect conditions')
    x = np.array([[[[float(lookup[c, a, p, e]['margin']) for e in ['happy', 'sad']]
                    for p in pairs[a]] for a in actors] for c in CONDITIONS])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite margin')
    samples = {r['sample_id']: r for r in rows if r['condition'] == 'clean'}
    old = {r['sample_id']: r for r in reference if r['condition'] == 'edge_decision_joint'}
    if samples.keys() != old.keys():
        raise ValueError('Historical reference sample mismatch')
    for r in rows:
        if r['sample_id'] not in samples or any(r[k] != samples[r['sample_id']][k] for k in ['actor', 'pair_id', 'statement', 'repetition', 'emotion']):
            raise ValueError('Metadata differs across conditions')
    checks = {
        'zero_noop_max': float(np.max(np.abs(x[1] - x[0]))),
        'clean_field_parity_max': max(abs(float(r['clean_margin']) - float(samples[r['sample_id']]['margin'])) for r in rows),
        'historical_clean_parity_max': max(abs(float(r['margin']) - float(old[s]['clean_margin'])) for s, r in samples.items()),
    }
    if any(not np.isfinite(v) or v > 1e-4 for v in checks.values()):
        raise ValueError(checks)
    draws = np.random.default_rng(20260915).integers(0, len(actors), (10000, len(actors)))
    counts = np.array([np.bincount(d, minlength=len(actors)) for d in draws])
    summary, contrasts, pairrows, statements = [], [], [], []
    estimates, bootstraps = {}, {}
    for ci, condition in enumerate(CONDITIONS):
        metrics = actor_metrics(x[ci], x[0])
        for metric, values in metrics.items():
            mean, boots = values.mean(), values[draws].mean(axis=1)
            estimates[condition, metric], bootstraps[condition, metric] = mean, boots
            summary.append(dict(condition=condition, metric=metric, **interval(mean, boots)))
        mean, boots = auc_samples(x[ci], counts)
        estimates[condition, 'roc_auc'], bootstraps[condition, 'roc_auc'] = mean, boots
        summary.append(dict(condition=condition, metric='roc_auc', **interval(mean, boots)))
        for ai, actor in enumerate(actors):
            for pi, pair in enumerate(pairs[actor]):
                meta = lookup[condition, actor, pair, 'happy']
                h, s = x[ci, ai, pi]
                bh, bs = x[0, ai, pi]
                pairrows.append(dict(condition=condition, actor=actor, pair_id=pair,
                    statement=meta['statement'], repetition=meta['repetition'],
                    happy_margin=h, sad_margin=s, happy_shift=h-bh, sad_shift=s-bs,
                    pair_separation=h-s, delta_pair_separation=(h-s)-(bh-bs), common_shift=((h-bh)+(s-bs))/2))
        for statement in sorted({r['statement'] for r in rows}):
            mask = [[lookup[condition, a, p, 'happy']['statement'] == statement for p in pairs[a]] for a in actors]
            if len({sum(m) for m in mask}) != 1 or not sum(mask[0]):
                raise ValueError('Balanced statement counts required')
            vals = np.array([x[ci, ai][mask[ai]] for ai in range(len(actors))])
            base = np.array([x[0, ai][mask[ai]] for ai in range(len(actors))])
            sm = {m: float(v.mean()) for m, v in actor_metrics(vals, base).items()}
            sm['roc_auc'] = auc_samples(vals, counts[:1])[0]
            statements.append(dict(condition=condition, statement=statement, n_samples=int(vals.size), **sm))
    for condition in CONDITIONS[1:]:
        for metric in [r['metric'] for r in summary if r['condition'] == 'clean']:
            contrasts.append(dict(condition=condition, metric=metric + '_minus_clean',
                **interval(estimates[condition, metric]-estimates['clean', metric],
                           bootstraps[condition, metric]-bootstraps['clean', metric])))
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary', summary), ('contrasts', contrasts), ('pair_shifts', pairrows), ('statement_descriptives', statements)]:
        write_csv(out / (name+'.csv'), data)
    (out/'verification.json').write_text(json.dumps(dict(n_samples=len(samples), n_rows=len(rows), n_actors=len(actors), checks=checks,
        score_ties={c: int((x[i] == 0).sum()) for i, c in enumerate(CONDITIONS)}), indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(dict(bootstrap='10000 paired actor-cluster draws, seed 20260915',
        auc='Cross-actor positive-negative comparisons recomputed with sampled actor multiplicities; ties=0.5',
        intervals='Unadjusted percentile 95%; all conditions share resamples',
        bootstrap_scope='Conditional on fitted leave-one-actor-out centering means; means are not reestimated in bootstrap draws, so calibration uncertainty is not included',
        decision='Happy if margin>0, otherwise sad; no threshold fitting or model calibration',
        statement_descriptives='Descriptive point estimates, not independent-actor evidence'), indent=2)+'\n')
    plot(summary, out)
    return summary, contrasts


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, metrics, title in [(axes[0], ['roc_auc'], 'Model ROC-AUC'),
                              (axes[1], ['pair_separation'], 'Matched happy−sad margin'),
                              (axes[2], ['happy_shift', 'sad_shift'], 'Margin shift from clean')]:
        for metric in metrics:
            rs = [r for r in summary if r['metric'] == metric]
            means = np.array([r['mean'] for r in rs])
            lo = np.array([r['ci_low'] for r in rs]); hi = np.array([r['ci_high'] for r in rs])
            # Percentile intervals need not contain the point estimate.
            positions = np.arange(len(rs))
            line, = ax.plot(positions, means, 'o-', label=metric)
            ax.vlines(positions, lo, hi, alpha=.7, color=line.get_color())
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=35, ha='right')
        ax.set_title(title); ax.grid(alpha=.2)
    axes[0].axhline(.5, color='gray', ls='--')
    axes[1].axhline(0, color='gray', ls='--'); axes[2].axhline(0, color='gray', ls='--'); axes[2].legend()
    fig.savefig(out/'natural_edge_centering.png', dpi=180)
    fig.savefig(out/'natural_edge_centering.pdf'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv', type=Path, required=True)
    p.add_argument('--reference-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    with args.input_csv.open() as f: rows = list(csv.DictReader(f))
    with args.reference_csv.open() as f: reference = list(csv.DictReader(f))
    analyze(rows, args.output_dir, reference)

if __name__ == '__main__':
    main()
