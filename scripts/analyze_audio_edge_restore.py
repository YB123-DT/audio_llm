#!/usr/bin/env python3
"""Compare audio-source value restoration restricted to query subsets."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
try:
    from scripts.analyze_residual_component_restore import grouped_pairs,estimate,write_csv
except ModuleNotFoundError:
    from analyze_residual_component_restore import grouped_pairs,estimate,write_csv

QUERIES=['decision','audio','end_marker','non_decision','all']


def summarize(rows):
    groups={q:[r for r in rows if r['condition']==f'edge_{q}_joint'] for q in QUERIES}
    if {r['condition'] for r in rows}!={f'edge_{q}_joint' for q in QUERIES}:
        raise ValueError('Missing or unexpected edge condition')
    pairs={q:grouped_pairs(g,'mediated_effect') for q,g in groups.items()}
    if any(p.keys()!=pairs['all'].keys() for p in pairs.values()):
        raise ValueError('Conditions must contain identical matched pairs')
    actors=sorted({a for a,_ in pairs['all']})
    draws=np.random.default_rng(20260915).integers(0,len(actors),(10000,len(actors)))
    summaries=[dict(query=q,metric=field,**estimate(grouped_pairs(groups[q],field),draws,actors))
               for q in QUERIES for field in ['patch_effect','remaining_effect','mediated_effect']]
    contrasts=[]
    for left,right in [('decision','all'),('decision','non_decision'),('audio','non_decision'),('end_marker','non_decision')]:
        contrasts.append(dict(contrast=f'{left}_minus_{right}',**estimate({k:pairs[left][k]-pairs[right][k] for k in pairs['all']},draws,actors)))
    contrasts.append(dict(contrast='all_minus_decision_minus_non_decision',**estimate({k:pairs['all'][k]-pairs['decision'][k]-pairs['non_decision'][k] for k in pairs['all']},draws,actors)))
    return summaries,contrasts


def verify(rows,reference):
    lookup={(r['sample_id'],r['condition']):r for r in rows}
    samples=sorted({r['sample_id'] for r in rows})
    old={r['sample_id']:r for r in reference if r['condition']=='v_audio_joint'}
    if set(old)!=set(samples):
        raise ValueError('Prior source-audio reference has different samples')
    errors={}
    for field in ['clean_margin','patched_margin','restored_margin']:
        errors[f'previous_audio_source_{field}_max_abs_error']=max(abs(float(lookup[s,'edge_all_joint'][field])-float(old[s][field])) for s in samples)
    errors['baseline_condition_consistency_max']=max(abs(float(lookup[s,f'edge_{q}_joint'][f])-float(lookup[s,'edge_all_joint'][f])) for s in samples for q in QUERIES for f in ['clean_margin','patched_margin'])
    if any(not np.isfinite(v) or v>1e-3 for v in errors.values()):
        raise ValueError(f'Edge reference parity failed: {errors}')
    return dict(n_samples=len(samples),n_rows=len(rows),checks=errors,tolerance=1e-3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv',type=Path,required=True)
    p.add_argument('--reference-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    with a.input_csv.open() as f: rows=list(csv.DictReader(f))
    with a.reference_csv.open() as f: reference=list(csv.DictReader(f))
    summary,contrasts=summarize(rows)
    verification=verify(rows,reference)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(a.output_dir/'edge_summary.csv',summary)
    write_csv(a.output_dir/'edge_contrasts.csv',contrasts)
    (a.output_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (a.output_dir/'analysis.json').write_text(json.dumps(dict(bootstrap_samples=10000,seed=20260915,unit='actor cluster',interval='unadjusted percentile 95%',interpretation='Conditional query-edge restore; effects not additive and equality not established by a null contrast'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in summary if r['metric']=='mediated_effect']
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.errorbar(range(5),[r['mean'] for r in chosen],yerr=[[r['mean']-r['ci_low'] for r in chosen],[r['ci_high']-r['mean'] for r in chosen]],fmt='o',capsize=4)
    ax.set(xticks=range(5),xticklabels=['decision','audio','end marker','non-decision','all'],ylabel='Removed donor-aligned margin effect',title='Audio source → selected queries: block18–23 edge restore')
    ax.axhline(0,color='gray',linewidth=.8);ax.grid(axis='y',alpha=.2)
    fig.savefig(a.output_dir/'audio_edge_restore.png',dpi=180)
    fig.savefig(a.output_dir/'audio_edge_restore.pdf');plt.close(fig)
    print(json.dumps(dict(effects=chosen,contrasts=contrasts),indent=2))

if __name__=='__main__':main()
