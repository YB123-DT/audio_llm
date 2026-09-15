#!/usr/bin/env python3
"""Fixed-protocol edge probes with explicitly transductive content corrections."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from scripts.analyze_edge_representation import folds, auc, write_csv
except ModuleNotFoundError:
    from analyze_edge_representation import folds, auc, write_csv


def fit_probe(x, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=1., max_iter=5000, solver='lbfgs', random_state=20260915).fit(scaler.transform(x), y)
    if clf.n_iter_.max() >= 5000:
        raise RuntimeError('Probe did not converge')
    w = clf.coef_[0] / scaler.scale_
    b = float(clf.intercept_[0] - w @ scaler.mean_)
    return w, b


def calibration_indices(rows, test):
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    held = set(actors[test])
    target = set(statements[test])
    if len(held) != 1 or len(target) != 1:
        raise ValueError('Joint calibration requires one held actor and statement')
    eligible = ~np.isin(actors, list(held))
    return np.flatnonzero(eligible & ~np.isin(statements, list(target))), np.flatnonzero(eligible & np.isin(statements, list(target)))


def project(x, offset):
    norm = np.linalg.norm(offset)
    if norm == 0:
        return x.copy()
    q = offset / norm
    return x - (x @ q)[:, None] * q


def metric_boot(y, score, actors, counts):
    unique = sorted(set(actors))
    ix = [np.flatnonzero(actors == a) for a in unique]
    h = [score[i][y[i] == 1] for i in ix]
    s = [score[i][y[i] == 0] for i in ix]
    kernel = np.array([[((hi[:, None] > sj) + .5 * (hi[:, None] == sj)).sum() for sj in s] for hi in h])
    acc = np.array([((score[i] > 0) == y[i]).mean() for i in ix])
    ba = counts @ acc / counts.sum(1)
    bu = np.einsum('bi,ij,bj->b', counts, kernel, counts) / ((counts @ np.array(list(map(len, h)))) * (counts @ np.array(list(map(len, s)))))
    return {'accuracy': (float(((score > 0) == y).mean()), ba), 'roc_auc': (auc(y, score), bu)}


def analyze(npz, metadata, reference, out, bootstrap=10000):
    import sklearn
    data = np.load(npz, allow_pickle=False)
    lookup = {r['sample_id']: r for r in csv.DictReader(Path(metadata).open())}
    rows = [lookup[str(s)] for s in data['sample_ids']]
    xall = data['edge_updates'] if 'edge_updates' in data else data['edge_vectors']
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    actors = np.array([r['actor'] for r in rows])
    for key in {(r['actor'], r['statement'], r['repetition']) for r in rows}:
        selected = [r for r in rows if (r['actor'], r['statement'], r['repetition']) == key]
        if len(selected) != 2 or {r['emotion'] for r in selected} != {'happy', 'sad'}:
            raise ValueError('Incomplete matched pair')
    nactor = len(set(actors))
    draws = np.random.default_rng(20260915).integers(nactor, size=(bootstrap, nactor))
    counts = np.array([np.bincount(v, minlength=nactor) for v in draws])
    old = {(int(r['layer']), r['regime'], r['sample_id']): float(r['score']) for r in csv.DictReader(Path(reference).open())}
    summaries, predictions, details, contrasts = [], [], [], []
    weights, names, offsets, source_means, target_means = [], [], [], [], []
    checks = dict(old_oof_parity_max=0., constant_shift_max=0., centered_pipeline_parity_max=0., source_only_noop_max=0., within_fold_auc_change_max=0.)
    for li, layer in enumerate(data['layers']):
        x = xall[:, li].astype(np.float64)
        for regime in ['statement', 'joint']:
            conditions = ['raw', 'source_center_noop', 'transductive_center'] if regime == 'statement' else ['raw', 'source_center_noop', 'independent_center', 'independent_rank1']
            oofs = {c: np.full(len(y), np.nan) for c in conditions}
            coverage = np.zeros(len(y), dtype=int)
            nfolds = 0
            fold_aucs = {c: [] for c in conditions}
            for fold, train, test in folds(rows, regime):
                coverage[test] += 1
                nfolds += 1
                w, b = fit_probe(x[train], y[train])
                raw = x[test] @ w + b
                meansource = x[train].mean(0)
                if regime == 'statement':
                    meantarget = x[test].mean(0)
                    cal_source, cal_target = train, test
                else:
                    cal_source, cal_target = calibration_indices(rows, test)
                    meansource = x[cal_source].mean(0)
                    meantarget = x[cal_target].mean(0)
                    assert not set(actors[test]) & set(actors[np.r_[cal_source, cal_target]])
                offset = meantarget - meansource
                scores = {'raw': raw}
                wc, bc = fit_probe(x[train] - meansource, y[train])
                noop = (x[test] - meansource) @ wc + bc
                checks['source_only_noop_max'] = max(checks['source_only_noop_max'], float(np.max(np.abs(noop - raw))))
                scores['source_center_noop'] = raw.copy()
                center_name = 'transductive_center' if regime == 'statement' else 'independent_center'
                scores[center_name] = raw - w @ offset
                centered = (x[test] - meantarget) @ wc + bc
                checks['centered_pipeline_parity_max'] = max(checks['centered_pipeline_parity_max'], float(np.max(np.abs(centered - scores[center_name]))))
                checks['constant_shift_max'] = max(checks['constant_shift_max'], float(np.ptp(scores[center_name] - raw)))
                checks['within_fold_auc_change_max'] = max(checks['within_fold_auc_change_max'], abs(auc(y[test], raw) - auc(y[test], scores[center_name])))
                condition_weights = {c: (w, b) for c in conditions}
                if regime == 'joint':
                    wp, bp = fit_probe(project(x[train], offset), y[train])
                    scores['independent_rank1'] = project(x[test], offset) @ wp + bp
                    condition_weights['independent_rank1'] = (project(wp[None], offset)[0], bp)
                oldscore = np.array([old[(int(layer), regime, rows[i]['sample_id'])] for i in test])
                checks['old_oof_parity_max'] = max(checks['old_oof_parity_max'], float(np.max(np.abs(raw - oldscore))))
                for condition, score in scores.items():
                    oofs[condition][test] = score
                    fa = auc(y[test], score)
                    fold_aucs[condition].append((actors[test][0] if regime == 'joint' else '', fa))
                    details.append(dict(layer=int(layer), regime=regime, fold=fold, condition=condition, n_train=len(train), n_test=len(test), n_cal_source=len(cal_source), n_cal_target=len(cal_target), accuracy=float(((score > 0) == y[test]).mean()), roc_auc=fa, offset_norm=float(np.linalg.norm(offset)), offset_probe_projection=float(w @ offset), score_shift_mean=float((score-raw).mean()), raw_intercept=b))
                    for i, s in zip(test, score):
                        predictions.append(dict(layer=int(layer), regime=regime, fold=fold, condition=condition, sample_id=rows[i]['sample_id'], actor=rows[i]['actor'], statement=rows[i]['statement'], emotion=rows[i]['emotion'], score=float(s)))
                    weights.append(condition_weights[condition][0]); names.append(f'{int(layer)}:{regime}:{fold}:{condition}')
                    offsets.append(offset); source_means.append(meansource); target_means.append(meantarget)
            if not np.all(coverage == 1):
                raise ValueError('Every sample must have exactly one held-out prediction')
            checks[f'layer_{layer}_{regime}_folds'] = nfolds
            checks[f'layer_{layer}_{regime}_coverage_min'] = int(coverage.min())
            checks[f'layer_{layer}_{regime}_coverage_max'] = int(coverage.max())
            base_metrics = metric_boot(y, oofs['raw'], actors, counts)
            for condition, score in oofs.items():
                if not np.isfinite(score).all():
                    raise ValueError('Incomplete OOF coverage')
                metrics = metric_boot(y, score, actors, counts)
                for metric, (mean, boot) in metrics.items():
                    lo, hi = np.quantile(boot, [.025, .975])
                    summaries.append(dict(layer=int(layer), regime=regime, condition=condition, metric=metric, mean=mean, ci_low=float(lo), ci_high=float(hi)))
                    delta = boot - base_metrics[metric][1]
                    lo, hi = np.quantile(delta, [.025, .975])
                    contrasts.append(dict(layer=int(layer), regime=regime, condition=condition, metric=metric, mean=mean-base_metrics[metric][0], ci_low=float(lo), ci_high=float(hi)))
                summaries.append(dict(layer=int(layer), regime=regime, condition=condition, metric='mean_within_fold_auc', mean=float(np.mean([a for _, a in fold_aucs[condition]])), ci_low='', ci_high=''))
    if checks['old_oof_parity_max'] > 1e-8 or checks['constant_shift_max'] > 1e-10 or checks['within_fold_auc_change_max'] != 0 or max(checks['centered_pipeline_parity_max'], checks['source_only_noop_max']) > 1e-5:
        raise ValueError(f'Parity checks failed: {checks}')
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [('summary', summaries), ('oof', predictions), ('folds', details), ('contrasts', contrasts)]:
        write_csv(out / f'counterfactual_{name}.csv', values)
    np.savez_compressed(out / 'counterfactual_directions.npz', weights=weights, names=names, offsets=offsets, source_means=source_means, target_means=target_means)
    (out / 'counterfactual_verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    (out / 'counterfactual_analysis.json').write_text(json.dumps(dict(n_samples=len(rows), bootstrap=bootstrap, seed=20260915, sklearn_version=sklearn.__version__, input_sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest(), method='Train-only StandardScaler + C=1 lbfgs; no tuning; per-layer independent. Raw-coordinate coefficients. Fixed-fitted OOF actor bootstrap; 95% percentile intervals unadjusted.', statement_center='Unlabeled transductive test-distribution centering; balanced known sample design. Equivalent to raw margin minus w dot(mean_target-mean_source); AUC invariant within fold, pooled AUC may change.', joint_calibration='Outer held actor never used for calibration. Other 23 actors provide both statement means WITHOUT target emotion labels; classifier uses only source statement labels. Calibration sees target statement distribution, so not strict unseen-statement induction.', rank1='Rank1 content axis estimated from same independent unlabeled calibration; both source training and target test projected then training-only scaler and classifier refit.', limitations='Mean shifts with class prevalence; balanced design matters. Centering success supports threshold contamination; failure does not prove rotation. Rank1 projection can remove emotion if axes overlap.'), indent=2)+'\n')


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((out / 'counterfactual_summary.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, regime in zip(axes, ['statement', 'joint']):
        conditions = ['raw', 'transductive_center'] if regime == 'statement' else ['raw', 'independent_center', 'independent_rank1']
        for condition in conditions:
            rs = [r for r in rows if r['regime'] == regime and r['condition'] == condition and r['metric'] == 'accuracy']
            ax.plot([int(r['layer']) for r in rs], [float(r['mean']) for r in rs], 'o-', label={'raw': 'Raw', 'transductive_center': 'Target-batch centering', 'independent_center': 'Independent centering', 'independent_rank1': 'Independent rank-1 projection'}[condition])
            ax.fill_between([int(r['layer']) for r in rs], [float(r['ci_low']) for r in rs], [float(r['ci_high']) for r in rs], alpha=.1)
        ax.axhline(.5, color='gray', ls=':')
        ax.set(title=('Statement transfer\n(target-batch centering)' if regime == 'statement' else 'Actor-held-out transfer\n(unlabeled target calibration)'), xlabel='Decoder block', ylabel='OOF accuracy')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / 'counterfactual_accuracy.png', dpi=180)
    fig.savefig(out / 'counterfactual_accuracy.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--representations', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--reference-oof', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    a = p.parse_args()
    analyze(a.representations, a.metadata, a.reference_oof, a.output_dir, a.bootstrap)
    plot(a.output_dir)
