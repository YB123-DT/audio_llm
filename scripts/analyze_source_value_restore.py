#!/usr/bin/env python3
"""Paired actor-cluster source-specific value restoration effects."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
try:
    from scripts.analyze_residual_component_restore import grouped_pairs, estimate, write_csv
except ModuleNotFoundError:
    from analyze_residual_component_restore import grouped_pairs, estimate, write_csv

SOURCES = ['audio','prompt','other','non_audio','decision','post_audio_marker','all']


def verify_sources(rows, reference_rows):
    lookup = {(r['sample_id'],r['condition']):r for r in rows}
    sample_ids = sorted({r['sample_id'] for r in rows})
    reference = {r['sample_id']:r for r in reference_rows if r['condition']=='v_joint'}
    if set(reference)!=set(sample_ids):
        raise ValueError('Prior all-V run and current run must have identical samples')
    def score(sample, source, field):
        return float(lookup[sample,f'v_{source}_joint'][field])
    checks = dict(
        prior_all_v_max_abs_error=max(abs(score(s,'all',field)-float(reference[s][field]))
            for s in sample_ids for field in ['clean_margin','patched_margin','restored_margin']),
        prompt_restore_no_effect_max=max(abs(score(s,'prompt','restored_margin')-score(s,'prompt','patched_margin')) for s in sample_ids),
        other_vs_non_audio_max=max(abs(score(s,'other','restored_margin')-score(s,'non_audio','restored_margin')) for s in sample_ids),
        baseline_condition_consistency_max=max(abs(score(s,source,field)-score(s,'all',field))
            for s in sample_ids for source in SOURCES for field in ['clean_margin','patched_margin']))
    if any(not np.isfinite(v) or v>1e-4 for v in checks.values()):
        raise ValueError(f'Source verification failed: {checks}')
    return dict(n_samples=len(sample_ids),n_rows=len(rows),numerical_checks=checks,tolerance=1e-4)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    expected = {f'v_{s}_joint' for s in SOURCES}
    if set(groups) != expected:
        raise ValueError(f'Conditions differ from fixed design: {set(groups)^expected}')
    pairs = {s: grouped_pairs(groups[f'v_{s}_joint'], 'mediated_effect') for s in SOURCES}
    if any(p.keys()!=pairs['all'].keys() for p in pairs.values()):
        raise ValueError('Source conditions have different pairs')
    actors = sorted({a for a,_ in pairs['all']})
    draws = np.random.default_rng(20260915).integers(0,len(actors),(10000,len(actors)))
    summaries = []
    for source in SOURCES:
        for field in ['patch_effect','remaining_effect','mediated_effect']:
            summaries.append(dict(source=source,metric=field,**estimate(grouped_pairs(groups[f'v_{source}_joint'],field),draws,actors)))
    contrasts=[]
    for left,right in [('audio','all'),('audio','other'),('audio','non_audio'),('other','decision'),('other','post_audio_marker')]:
        diff = {k:pairs[left][k]-pairs[right][k] for k in pairs['all']}
        contrasts.append(dict(contrast=f'{left}_minus_{right}',**estimate(diff,draws,actors)))
    interaction={k:pairs['all'][k]-sum(pairs[s][k] for s in ['audio','prompt','other']) for k in pairs['all']}
    contrasts.append(dict(contrast='all_minus_partition_sum',**estimate(interaction,draws,actors)))
    return summaries,contrasts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reference-csv',type=Path,required=True,help='Previous all-source QK/V experiment CSV')
    args=parser.parse_args()
    with args.input_csv.open() as f: rows=list(csv.DictReader(f))
    summaries,contrasts=summarize(rows)
    with args.reference_csv.open() as f: reference=list(csv.DictReader(f))
    verification=verify_sources(rows,reference)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    write_csv(args.output_dir/'source_summary.csv',summaries)
    write_csv(args.output_dir/'source_contrasts.csv',contrasts)
    (args.output_dir/'analysis.json').write_text(json.dumps(dict(bootstrap_samples=10000,seed=20260915,unit='actor cluster',effect='donor_sign*(patched_margin-restored_margin), averaged within pair',scope='source-specific V at all queries; not edge-specific mediation',equivalence='No equivalence margin specified; confidence interval crossing zero does not establish equality'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in summaries if r['metric']=='mediated_effect']
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    ax.errorbar(range(len(chosen)),[r['mean'] for r in chosen],yerr=[[r['mean']-r['ci_low'] for r in chosen],[r['ci_high']-r['mean'] for r in chosen]],fmt='o',capsize=4)
    labels=['audio','prompt','other','non-audio','decision','end marker','all']
    ax.set(xticks=range(len(chosen)),xticklabels=labels,ylabel='Removed donor-aligned margin effect',title='Block18–23 source-V restore after block17 audio swap')
    ax.axhline(0,color='gray',linewidth=.8);ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output_dir/'source_value_restore.png',dpi=180)
    fig.savefig(args.output_dir/'source_value_restore.pdf');plt.close(fig)
    print(json.dumps(dict(summaries=chosen,contrasts=contrasts),indent=2))

if __name__=='__main__':
    main()
