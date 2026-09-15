#!/usr/bin/env python3
"""Single-block and cumulative direct-edge restoration."""
import argparse,csv,json
from pathlib import Path
import numpy as np
try:
    from scripts.analyze_residual_component_restore import grouped_pairs,estimate,write_csv
    from scripts.run_block_direct_edge_restore import restore_schedules
except ModuleNotFoundError:
    from analyze_residual_component_restore import grouped_pairs,estimate,write_csv
    from run_block_direct_edge_restore import restore_schedules


def summarize(rows):
    schedules=restore_schedules()
    if {r['condition'] for r in rows}!=set(schedules):raise ValueError('Unexpected schedule set')
    groups={s:[r for r in rows if r['condition']==s] for s in schedules}
    pairs={s:grouped_pairs(rs,'mediated_effect') for s,rs in groups.items()}
    first=next(iter(pairs.values()))
    if any(p.keys()!=first.keys() for p in pairs.values()):raise ValueError('Pair mismatch')
    actors=sorted({a for a,_ in first})
    draws=np.random.default_rng(20260915).integers(0,len(actors),(10000,len(actors)))
    summary=[dict(condition=s,metric=f,**estimate(grouped_pairs(rs,f),draws,actors)) for s,rs in groups.items() for f in ['patch_effect','remaining_effect','mediated_effect']]
    increments=[];previous='direct_block_18'
    for end in range(19,24):
        current=f'direct_cumulative_18_{end}'
        increments.append(dict(added_block=end,**estimate({k:pairs[current][k]-pairs[previous][k] for k in first},draws,actors)))
        previous=current
    return summary,increments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--reference-csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    with a.input_csv.open() as f:rows=list(csv.DictReader(f))
    with a.reference_csv.open() as f:old={r['sample_id']:r for r in csv.DictReader(f) if r['condition']=='edge_decision_joint'}
    summary,inc=summarize(rows)
    joint={r['sample_id']:r for r in rows if r['condition']=='direct_cumulative_18_23'}
    if joint.keys()!=old.keys():raise ValueError('Historical sample mismatch')
    err=max(abs(float(r[k])-float(old[s][k])) for s,r in joint.items() for k in ['clean_margin','patched_margin','restored_margin'])
    baselineerr=max(abs(float(r[k])-float(joint[r['sample_id']][k])) for r in rows for k in ['clean_margin','patched_margin'])
    if not np.isfinite(err) or err>1e-3 or baselineerr>1e-4:raise ValueError('Reference parity failed')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(a.output_dir/'block_summary.csv',summary);write_csv(a.output_dir/'cumulative_increments.csv',inc)
    (a.output_dir/'verification.json').write_text(json.dumps(dict(n_rows=len(rows),n_samples=len(joint),historical_joint_max_error=err,baseline_condition_max_error=baselineerr,bootstrap='10000 paired actor resamples numpy seed20260915; unadjusted95%CI'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for kind in ['single','cumulative']:
        keys=[f'direct_block_{l}' if kind=='single' or l==18 else f'direct_cumulative_18_{l}' for l in range(18,24)]
        rs=[next(r for r in summary if r['condition']==k and r['metric']=='mediated_effect') for k in keys]
        ax.errorbar(range(18,24),[r['mean'] for r in rs],yerr=[[r['mean']-r['ci_low'] for r in rs],[r['ci_high']-r['mean'] for r in rs]],fmt='o-',capsize=3,label=kind)
    ax.set(xlabel='Block / cumulative endpoint',ylabel='Removed donor-aligned effect',title='Direct audio→decision restore after post17 donor patch');ax.axhline(0,color='gray',ls='--');ax.legend();ax.grid(alpha=.2)
    fig.savefig(a.output_dir/'block_restore.png',dpi=180);fig.savefig(a.output_dir/'block_restore.pdf');plt.close(fig)

if __name__=='__main__':main()
