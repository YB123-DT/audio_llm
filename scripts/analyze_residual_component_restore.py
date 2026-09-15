#!/usr/bin/env python3
"""Summarize donor-aligned residual and conditional component effects."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def grouped_pairs(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['actor'], row['pair_id'])].append(row)
    result = {}
    for key, group in groups.items():
        if len(group) != 2 or {r['emotion'] for r in group} != {'happy', 'sad'}:
            raise ValueError(f'Incomplete/duplicate pair {key}')
        values = [float(r[field]) for r in group]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite pair {key}')
        result[key] = float(np.mean(values))
    return result


def estimate(pairs, draws, actors):
    values = np.array([np.mean([v for (a, _), v in pairs.items() if a == actor]) for actor in actors])
    samples = values[draws].mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {'mean': float(values.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
            'n_pairs': len(pairs), 'n_actors': len(actors)}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def analyze(input_path, attribution_path, output):
    with input_path.open() as f:
        rows = list(csv.DictReader(f))
    with attribution_path.open() as f:
        attribution = list(csv.DictReader(f))
    actors = sorted({r['actor'] for r in rows})
    draws = np.random.default_rng(20260915).integers(0, len(actors), (10000, len(actors)))
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    summaries, pairs_by_condition = [], {}
    expected_keys = None
    for condition, group in groups.items():
        pairs_by_condition[condition] = grouped_pairs(group, 'mediated_effect')
        keys = pairs_by_condition[condition].keys()
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f'Missing pairs in {condition}')
        for field in ['patch_effect', 'remaining_effect', 'mediated_effect']:
            summaries.append({'condition': condition, 'metric': field,
                              **estimate(grouped_pairs(group, field), draws, actors)})
    contrasts = []
    for suffix in ['18', '19', '20', '21', '22', '23', 'joint']:
        a, m = f'attention_{suffix}', f'mlp_{suffix}'
        if a in pairs_by_condition and m in pairs_by_condition:
            pa, pm = pairs_by_condition[a], pairs_by_condition[m]
            if pa.keys() != pm.keys():
                raise ValueError('Unmatched contrast')
            contrasts.append({'condition': suffix, 'metric': 'attention_minus_mlp_mediation',
                              **estimate({k: pa[k]-pm[k] for k in pa}, draws, actors)})
        q, v = f'qk_{suffix}', f'v_{suffix}'
        if q in pairs_by_condition and v in pairs_by_condition:
            pq, pv = pairs_by_condition[q], pairs_by_condition[v]
            contrasts.append({'condition': suffix, 'metric': 'qk_minus_v_mediation',
                              **estimate({k: pq[k]-pv[k] for k in pq}, draws, actors)})
    lens_groups = defaultdict(list)
    for row in attribution:
        lens_groups[int(row['layer_index'])].append(row)
    lens = []
    for layer, group in sorted(lens_groups.items()):
        if grouped_pairs(group, 'delta_raw').keys() != expected_keys:
            raise ValueError(f'Missing attribution pairs in layer {layer}')
        for field in ['delta_raw', 'delta_lens']:
            lens.append({'layer_index': layer, 'metric': field,
                         **estimate(grouped_pairs(group, field), draws, actors)})
        # Natural clean H-S has a separate interpretation from intervention effects.
        natural = [dict(r, natural=float(r['clean_lens']) * (2 if r['emotion']=='happy' else -2)) for r in group]
        lens.append({'layer_index': layer, 'metric': 'natural_clean_happy_minus_sad_lens',
                     **estimate(grouped_pairs(natural, 'natural'), draws, actors)})
    att = next(r for r in summaries if r['condition']=='attention_joint' and r['metric']=='mediated_effect')
    contrast = next(r for r in contrasts if r['condition']=='joint' and r['metric']=='attention_minus_mlp_mediation')
    gate = {'attention_wins': att['ci_low']>0 and contrast['ci_low']>0,
            'joint_attention_mediation': att, 'joint_attention_minus_mlp': contrast,
            'bootstrap': '10000 paired actor-cluster resamples; numpy seed 20260915',
            'scope': 'Conditional decision-position component restoration after block17 audio swap; exploratory block scans have unadjusted intervals.'}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'component_summary.csv', summaries)
    write_csv(output/'component_contrasts.csv', contrasts)
    write_csv(output/'residual_summary.csv', lens)
    (output/'attention_gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle('Post-block17 audio swap: 96 pairs, 24 actor-cluster bootstrap intervals')
    for axis, metric, title in [(ax[0], 'delta_raw', 'Raw residual projection'), (ax[1], 'delta_lens', 'Final-norm logit lens')]:
        rs = [r for r in lens if r['metric']==metric]
        axis.errorbar([r['layer_index'] for r in rs], [r['mean'] for r in rs],
                      yerr=[[r['mean']-r['ci_low'] for r in rs], [r['ci_high']-r['mean'] for r in rs]], fmt='o-', markersize=3)
        axis.set(xlabel='Block output (zero-based)', ylabel='Donor-aligned change', title=title)
    for component, offset in [('attention', -.10), ('mlp', .10)]:
        rs = [next(r for r in summaries if r['condition']==f'{component}_{s}' and r['metric']=='mediated_effect') for s in ['18','19','20','21','22','23','joint']]
        ax[2].errorbar(np.arange(7)+offset, [r['mean'] for r in rs], yerr=[[r['mean']-r['ci_low'] for r in rs], [r['ci_high']-r['mean'] for r in rs]], fmt='o', label=component, capsize=2)
    ax[2].set(xticks=range(7), xticklabels=['18','19','20','21','22','23','joint'], ylabel='Removed patch effect', title='Decision component restore')
    ax[2].legend()
    for axis in ax:
        axis.axhline(0, color='gray', lw=.8)
        axis.grid(alpha=.2)
    fig.savefig(output/'residual_component.png', dpi=180)
    fig.savefig(output/'residual_component.pdf')
    plt.close(fig)
    if 'qk_joint' in pairs_by_condition:
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        suffixes = [s for s in ['18','19','20','21','22','23','joint'] if f'qk_{s}' in pairs_by_condition]
        for kind, offset in [('qk', -.10), ('v', .10)]:
            rs = [next(r for r in summaries if r['condition']==f'{kind}_{s}' and r['metric']=='mediated_effect') for s in suffixes]
            axis.errorbar(np.arange(len(rs))+offset, [r['mean'] for r in rs],
                          yerr=[[r['mean']-r['ci_low'] for r in rs], [r['ci_high']-r['mean'] for r in rs]],
                          fmt='o', capsize=3, label=kind.upper())
        axis.set(xticks=range(len(suffixes)), xticklabels=suffixes,
                 ylabel='Removed donor-aligned margin effect',
                 title='After block17 audio swap: decision Q + all-source K vs all-source V')
        axis.axhline(0, color='gray', lw=.8)
        axis.grid(alpha=.2)
        axis.legend()
        fig.savefig(output/'qkv_restore.png', dpi=180)
        fig.savefig(output/'qkv_restore.pdf')
        plt.close(fig)
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv', type=Path, required=True)
    p.add_argument('--attribution-csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    analyze(a.input_csv, a.attribution_csv, a.output_dir)
