#!/usr/bin/env python3
"""Actor-cluster uncertainty over a24-layer causal sweep."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(rows, n_boot=10000):
    actors=sorted({r['actor'] for r in rows})
    if {int(r['layer']) for r in rows} != set(range(24)):raise ValueError('Need layers0–23')
    grouped={}
    for r in rows:
        key=(r['actor'],r['pair_id'],int(r['layer']))
        if r['emotion'] in grouped.setdefault(key,{}):raise ValueError('Duplicate emotion')
        grouped[key][r['emotion']]=float(r['patch_effect'])
    if any(set(v)!={'happy','sad'} for v in grouped.values()):raise ValueError('Incomplete pair')
    pair_keys=sorted({(a,p) for a,p,l in grouped})
    if any((a,p,l) not in grouped for a,p in pair_keys for l in range(24)):raise ValueError('Missing layer')
    actor_values=np.array([[np.mean([np.mean(list(grouped[(a,p,l)].values())) for aa,p in pair_keys if aa==a]) for l in range(24)] for a in actors])
    if not np.isfinite(actor_values).all():raise ValueError('Nonfinite effects')
    counts=[sum(a==aa for aa,p in pair_keys) for a in actors]
    if len(set(counts))!=1:raise ValueError('Unbalanced actors')
    draws=np.random.default_rng(20260915).integers(0,len(actors),(n_boot,len(actors)))
    mean=actor_values.mean(0);boot=actor_values[draws].mean(1)
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    radius=float(np.quantile(np.max(np.abs(boot-mean),axis=1),.95))
    summary=[dict(layer=l,mean=float(mean[l]),ci_low=float(lo[l]),ci_high=float(hi[l]),simultaneous_low=float(mean[l]-radius),simultaneous_high=float(mean[l]+radius)) for l in range(24)]
    contrast=boot-boot[:,17,None]
    cm=mean-mean[17];cl,ch=np.quantile(contrast,[.025,.975],axis=0)
    cr=float(np.quantile(np.max(np.abs(contrast-cm),axis=1),.95))
    contrasts=[dict(layer=l,mean=float(cm[l]),ci_low=float(cl[l]),ci_high=float(ch[l]),simultaneous_low=float(cm[l]-cr),simultaneous_high=float(cm[l]+cr)) for l in range(24)]
    return summary,contrasts,dict(n_actors=len(actors),n_pairs=len(pair_keys),bootstrap_replicates=n_boot,seed=20260915,simultaneous_method='95th percentile max absolute centered bootstrap deviation across 24 layers; unstudentized actor-cluster bootstrap',simultaneous_radius=radius,contrast_simultaneous_radius=cr,peak_point_estimate_layer=int(np.argmax(mean)))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--reference-csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    with a.input_csv.open() as f:rows=list(csv.DictReader(f))
    with a.reference_csv.open() as f:old={r['sample_id']:r for r in csv.DictReader(f) if r['condition']=='edge_decision_joint'}
    current={r['sample_id']:r for r in rows if int(r['layer'])==17}
    if current.keys()!=old.keys():raise ValueError('Reference sample mismatch')
    parity=max(abs(float(r[k])-float(old[s][k])) for s,r in current.items() for k in ['clean_margin','patched_margin'])
    if parity>1e-3 or not np.isfinite(parity):raise ValueError('Reference parity failure')
    baseline_error=max(abs(float(r['clean_margin'])-float(current[r['sample_id']]['clean_margin'])) for r in rows)
    final_error=max(abs(float(r['patch_effect'])) for r in rows if int(r['layer'])==23)
    if baseline_error>1e-6 or final_error>1e-6:raise ValueError('Baseline invariance or final-layer structural zero failed')
    summary,contrasts,meta=summarize(rows)
    meta.update(n_rows=len(rows),layer17_historical_max_error=parity,baseline_across_layers_max_error=baseline_error,final_layer_effect_max=final_error)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write(a.output_dir/'sweep_summary.csv',summary);write(a.output_dir/'sweep_vs17.csv',contrasts)
    (a.output_dir/'verification.json').write_text(json.dumps(meta,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    layers=np.arange(24);means=[r['mean'] for r in summary]
    ax.fill_between(layers,[r['simultaneous_low'] for r in summary],[r['simultaneous_high'] for r in summary],alpha=.15,label='95% simultaneous bootstrap band')
    ax.errorbar(layers,means,yerr=[[r['mean']-r['ci_low'] for r in summary],[r['ci_high']-r['mean'] for r in summary]],fmt='o-',capsize=3,label='Mean CE; pointwise 95% CI')
    ax.axhline(0,color='gray',ls='--');ax.axvline(17,color='gray',alpha=.4);ax.set(xlabel='Post-block audio patch layer (0-based)',ylabel='Donor-aligned margin effect',title='Full matched audio causal sweep');ax.set_xticks(range(24));ax.legend();ax.grid(alpha=.2)
    fig.savefig(a.output_dir/'sweep.png',dpi=180);fig.savefig(a.output_dir/'sweep.pdf');plt.close(fig)

if __name__=='__main__':main()
