#!/usr/bin/env python3
"""Natural edge contribution and gain, with paired actor-cluster uncertainty."""
import argparse,csv,json
from pathlib import Path
import numpy as np
try:
    from scripts.analyze_residual_component_restore import write_csv
except ModuleNotFoundError:
    from analyze_residual_component_restore import write_csv

ALPHAS=[0.,.5,1.,1.5,2.]


def auc_samples(values,counts):
    # values: actor,pair,emotion(H,S). Resampling actors duplicates both classes.
    h,s=values[:,:,0],values[:,:,1]
    wins=(h[:,None,:,None]>s[None,:,None,:]).astype(float)+.5*(h[:,None,:,None]==s[None,:,None,:])
    kernel=wins.sum(axis=(2,3))
    denominator=(counts.sum(axis=1)*h.shape[1])**2
    return float(kernel.sum()/h.size**2),np.einsum('bi,ij,bj->b',counts,kernel,counts)/denominator


def interval(mean,boots):
    lo,hi=np.quantile(boots,[.025,.975])
    return dict(mean=float(mean),ci_low=float(lo),ci_high=float(hi))


def analyze(rows,out,reference):
    actors=sorted({r['actor'] for r in rows})
    pairs={a:sorted({r['pair_id'] for r in rows if r['actor']==a}) for a in actors}
    if len({len(p) for p in pairs.values()})!=1:raise ValueError('Balanced actor pairs required')
    lookup={(float(r['alpha']),r['actor'],r['pair_id'],r['emotion']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate rows')
    if {float(r['alpha']) for r in rows}!=set(ALPHAS):raise ValueError('Wrong alphas')
    x=np.array([[[[float(lookup[alpha,a,p,e]['margin']) for e in ['happy','sad']] for p in pairs[a]] for a in actors] for alpha in ALPHAS])
    if not np.isfinite(x).all():raise ValueError('Nonfinite scores')
    baseline=x[2]
    samples={r['sample_id']:r for r in rows if float(r['alpha'])==1}
    old={r['sample_id']:r for r in reference if r['condition']=='edge_decision_joint'}
    if samples.keys()!=old.keys():raise ValueError('Reference samples differ')
    checks=dict(alpha1_noop_max=max(abs(float(r['margin'])-float(r['clean_margin'])) for r in samples.values()),old_clean_parity_max=max(abs(float(r['clean_margin'])-float(old[s]['clean_margin'])) for s,r in samples.items()),clean_across_alpha_max=max(abs(float(r['clean_margin'])-float(samples[r['sample_id']]['clean_margin'])) for r in rows))
    if any(not np.isfinite(v) or v>1e-4 for v in checks.values()):raise ValueError(checks)
    draws=np.random.default_rng(20260915).integers(0,len(actors),(10000,len(actors)))
    counts=np.array([np.bincount(d,minlength=len(actors)) for d in draws])
    summary=[];boot_store={};mean_store={}
    def add(alpha,metric,actor_values):
        boots=actor_values[draws].mean(axis=1);mean=actor_values.mean()
        summary.append(dict(alpha=alpha,metric=metric,**interval(mean,boots)))
        boot_store[alpha,metric]=boots;mean_store[alpha,metric]=mean
    for i,alpha in enumerate(ALPHAS):
        a=x[i];gap=a[:,:,0]-a[:,:,1];shift=a-baseline
        add(alpha,'accuracy',np.stack([a[:,:,0]>0,a[:,:,1]<=0],axis=-1).mean(axis=(1,2)))
        add(alpha,'happy_prediction_fraction',(a>0).mean(axis=(1,2)))
        add(alpha,'pair_direction_fraction',(gap>0).mean(axis=1))
        add(alpha,'pair_separation',gap.mean(axis=1))
        add(alpha,'happy_margin',a[:,:,0].mean(axis=1));add(alpha,'sad_margin',a[:,:,1].mean(axis=1))
        add(alpha,'common_shift',shift.mean(axis=(1,2)))
        add(alpha,'happy_shift',shift[:,:,0].mean(axis=1));add(alpha,'sad_shift',shift[:,:,1].mean(axis=1))
        mean,boots=auc_samples(a,counts)
        summary.append(dict(alpha=alpha,metric='roc_auc',**interval(mean,boots)))
        boot_store[alpha,'roc_auc']=boots;mean_store[alpha,'roc_auc']=mean
    contrasts=[]
    for alpha in ALPHAS:
        if alpha==1:continue
        for metric in ['roc_auc','pair_separation','accuracy','pair_direction_fraction']:
            contrasts.append(dict(alpha=alpha,metric=metric+'_minus_alpha1',**interval(mean_store[alpha,metric]-mean_store[1,metric],boot_store[alpha,metric]-boot_store[1,metric])))
    c=baseline-x[0];dc=c[:,:,0]-c[:,:,1]
    necessity=[]
    for name,v in [('C_happy',c[:,:,0]),('C_sad',c[:,:,1]),('delta_C',dc),('positive_delta_C_fraction',(dc>0).astype(float)),('negative_delta_C_fraction',(dc<0).astype(float))]:
        av=v.mean(axis=1);necessity.append(dict(metric=name,**interval(av.mean(),av[draws].mean(axis=1))))
    pairrows=[]
    for ai,a in enumerate(actors):
        for pi,pair in enumerate(pairs[a]):
            r=lookup[1.,a,pair,'happy']
            pairrows.append(dict(pair_id=pair,actor=a,statement=r['statement'],repetition=r['repetition'],C_happy=c[ai,pi,0],C_sad=c[ai,pi,1],delta_C=dc[ai,pi],positive=bool(dc[ai,pi]>0)))
    grouped=[]
    for field in ['actor','statement']:
        for group in sorted({r[field] for r in pairrows}):
            vals=[r['delta_C'] for r in pairrows if r[field]==group]
            grouped.append(dict(group_type=field,group=group,n_pairs=len(vals),mean_delta_C=float(np.mean(vals)),positive_fraction=float(np.mean(np.array(vals)>0))))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('gain_summary',summary),('gain_contrasts',contrasts),('necessity_summary',necessity),('necessity_pairs',pairrows),('necessity_context',grouped)]:write_csv(out/(name+'.csv'),data)
    (out/'verification.json').write_text(json.dumps(dict(n_samples=len(samples),n_rows=len(rows),n_actors=len(actors),checks=checks,score_ties={str(alpha):int((x[i]==0).sum()) for i,alpha in enumerate(ALPHAS)}),indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(dict(bootstrap='10000 paired actor-cluster resamples, numpy seed20260915',auc='Recomputed using cross-actor positive-negative concordance and resampled actor multiplicities; ties=0.5',intervals='Unadjusted percentile95; no optimized bias, threshold or gain'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for axis,metrics,title in [(axes[0],['roc_auc'],'ROC-AUC'),(axes[1],['pair_separation'],'Matched happy−sad margin'),(axes[2],['happy_shift','sad_shift'],'Shift relative to alpha=1')]:
        for metric in metrics:
            rs=[r for r in summary if r['metric']==metric]
            axis.errorbar(ALPHAS,[r['mean'] for r in rs],yerr=[[r['mean']-r['ci_low'] for r in rs],[r['ci_high']-r['mean'] for r in rs]],fmt='o-',capsize=3,label=metric)
        axis.set(title=title,xlabel='Natural audio→decision gain alpha');axis.grid(alpha=.2)
    axes[0].axhline(.5,color='gray',ls='--');axes[1].axhline(0,color='gray',ls='--');axes[2].legend()
    fig.savefig(out/'clean_edge_gain.png',dpi=180);fig.savefig(out/'clean_edge_gain.pdf');plt.close(fig)
    print(json.dumps(necessity,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--reference-csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    with a.input_csv.open() as f:rows=list(csv.DictReader(f))
    with a.reference_csv.open() as f:ref=list(csv.DictReader(f))
    analyze(rows,a.output_dir,ref)

if __name__=='__main__':main()
