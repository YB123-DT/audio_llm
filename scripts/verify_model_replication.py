#!/usr/bin/env python3
"""Independent frozen-cohort artifact audit; no probe fits or project imports."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

C = ('clean','no_attn','no_mlp')
E = ('block6','final_answer')
COHORTS = {
 'ravdess': ('artifacts/ravdess_happy_sad_intensity01.csv',192,24,2,2,'c1894d6dc765947521254a62ef7cc582519b0110acc52e6a6225be9806bf74d6'),
 'crema-d': ('reports/2026-09-16-crema-confirmation/manifest.csv',1936,88,11,1,'b7db3dff0af5a0e18a435ebbe888181d001501a190f4bba9f937ed2c8c6993a3'),
}
def read(path):
 with Path(path).open() as f: return list(csv.DictReader(f))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
def close(a,b):
 assert np.isclose(float(a),float(b),rtol=0,atol=1e-12),(a,b)
def audit(dataset, root):
 manifest,n,na,ns,nr,expected_sha=COHORTS[dataset]
 assert sha(manifest)==expected_sha
 rows=read(manifest); ids=[r['sample_id'] for r in rows]; lookup={r['sample_id']:r for r in rows}
 actors=sorted({r['actor'] for r in rows}); stmts=sorted({r['statement'] for r in rows}); reps=sorted({r['repetition'] for r in rows})
 assert len(rows)==len(lookup)==n and (len(actors),len(stmts),len(reps))==(na,ns,nr)
 assert len({r['intensity'] for r in rows})==1
 assert {(r['actor'],r['statement'],r['repetition'],r['emotion']) for r in rows}=={(a,s,r,e) for a in actors for s in stmts for r in reps for e in ('happy','sad')}
 out=root/dataset; states=out/'intervention_states.npz'; info=json.loads(states.with_suffix('.json').read_text()); analysis=out/'analysis'
 assert info['model']=='LLaMA-Omni2-0.5B' and info['dataset']==dataset and info['n_samples']==n
 assert info['conditions']==list(C) and info['target_blocks']==list(range(1,7)) and info['decoder_forwards']==3*n
 digest=sha(states); assert digest==info['output_sha256']
 with np.load(states,allow_pickle=False) as z:
  assert z['sample_ids'].tolist()==ids and tuple(z['conditions'])==C
  for e in E: assert z[e].shape==(n,3,896) and np.isfinite(z[e]).all()
 checks=info['checks']; assert checks['nonanswer_block_comparisons']==48*n and checks['suppressed_update_calls']==12*n
 assert checks['unhooked_clean_final_max']==0
 parity=checks['official_forward_parity']; assert parity['final_state_max_abs_error']==0 and 0<=parity['last_logit_max_abs_error']<=1e-4
 runtime=info['runtime']; assert runtime['model_parameters_frozen'] is True and runtime['final_checkpoint_alias_audit'] is True
 assert runtime['checkpoint_sha256']=='a34de68dc82e051d01f4c6e32c43addaf6f52721cc4e89db06868b616b76fc5c'
 assert runtime['whisper_sha256']=='e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb'
 assert runtime['source_commit']=='c8afa9061a9c2d2c1919f7293f5492d946869752'
 assert runtime['model_revision']=='a16aa9a4ea3f2f363c3db728e8e83ee08e60922c'
 meta=json.loads((analysis/'analysis.json').read_text()); assert meta['model']==info['model'] and meta['dataset']==dataset
 assert meta['n_probe_fits']==6*na*ns and meta['clean_probe_fits_reused']==0 and meta['seed']==20260915 and meta['bootstrap']==10000
 for p,d in meta['sources_sha256'].items(): assert sha(p)==d, p
 # Reconstruct every expected fold membership, without analysis helpers.
 expected_members=set()
 for a in actors:
  for s in stmts:
   fold=f'actor_{a}_statement_{s}'
   for r in rows:
    if r['statement']==s: expected_members.add((fold,'test' if r['actor']==a else 'train',r['sample_id']))
 membership=read(analysis/'fold_membership.csv'); actual=[(r['fold'],r['subset'],r['sample_id']) for r in membership]
 assert len(actual)==len(set(actual)) and set(actual)==expected_members
 pred=read(analysis/'oof_predictions.csv'); assert len(pred)==6*n
 scores={}
 for r in pred:
  k=(r['endpoint'],r['condition'],r['sample_id']); assert k not in scores
  m=lookup[r['sample_id']]; assert all(r[x]==m[x] for x in ('actor','statement','repetition','emotion'))
  assert r['fold']==f"actor_{m['actor']}_statement_{m['statement']}"
  score=float(r['score']); assert np.isfinite(score) and r['prediction']==('happy' if score>0 else 'sad'); scores[k]=score
 assert set(scores)=={(e,c,i) for e in E for c in C for i in ids}
 values={}
 for e in E:
  for c in C:
   matrix=np.empty((na,ns))
   for ai,a in enumerate(actors):
    for si,s in enumerate(stmts):
     rr=[r for r in rows if r['actor']==a and r['statement']==s]
     pos=[scores[e,c,r['sample_id']] for r in rr if r['emotion']=='happy']; neg=[scores[e,c,r['sample_id']] for r in rr if r['emotion']=='sad']
     assert len(pos)==len(neg)==nr
     matrix[ai,si]=sum(float(h>t)+.5*float(h==t) for h in pos for t in neg)/(nr*nr)
   values[e,c]=matrix
 fold_metrics=read(analysis/'fold_metrics.csv'); seen=set()
 for r in fold_metrics:
  key=(r['endpoint'],r['condition'],r['actor'],r['statement']); assert key not in seen; seen.add(key)
  assert int(r['n_train'])==2*nr*(na-1) and int(r['n_test'])==2*nr
  close(r['roc_auc'],values[key[:2]][actors.index(r['actor']),stmts.index(r['statement'])])
 assert len(seen)==6*na*ns
 # Sample actor indices directly; unlike production no bootstrap-count matrix.
 draws=np.random.default_rng(20260915).integers(0,na,(10000,na))
 estimates={}
 for e in E:
  for s in stmts+['combined']:
   for c in C:
    vector=values[e,c].mean(axis=1) if s=='combined' else values[e,c][:,stmts.index(s)]
    estimates[e,s,c]=(vector.mean(),vector[draws].mean(axis=1))
 summaries=[r for r in read(analysis/'summary.csv') if r['metric']=='mean_within_fold_auc']
 assert len(summaries)==6*(ns+1)
 assert {(r['endpoint'],r['statement'],r['condition']) for r in summaries}==set(estimates)
 for r in summaries:
  mean,dd=estimates[r['endpoint'],r['statement'],r['condition']]; lo,hi=np.quantile(dd,[.025,.975])
  for k,v in [('mean',mean),('ci_low',lo),('ci_high',hi)]: close(r[k],v)
 contrast_defs={'delta_A':('no_attn','clean'),'delta_A_minus_M':('no_attn','no_mlp'),'delta_M_secondary':('no_mlp','clean')}
 contrasts=[r for r in read(analysis/'contrasts.csv') if r['metric']=='mean_within_fold_auc']; assert len(contrasts)==6*(ns+1)
 assert len({(r['endpoint'],r['statement'],r['contrast']) for r in contrasts})==len(contrasts)
 for r in contrasts:
  l,rr=contrast_defs[r['contrast']]; assert (r['left'],r['right'])==(l,rr)
  lm,ld=estimates[r['endpoint'],r['statement'],l]; rm,rd=estimates[r['endpoint'],r['statement'],rr]; lo,hi=np.quantile(ld-rd,[.025,.975])
  for k,v in [('mean',lm-rm),('ci_low',lo),('ci_high',hi)]: close(r[k],v)
 result=dict(status='PASS', audit_script_sha256=sha(__file__),dataset=dataset,n_samples=n,n_actors=na,n_statements=ns,n_repetitions=nr,manifest_sha256=expected_sha,states_sha256=digest,nonanswer_checks=48*n,suppression_checks=12*n,oof_records=6*n,fold_membership_rows=len(membership),independently_recomputed_auc_summaries=len(summaries),independently_recomputed_paired_auc_contrasts=len(contrasts),bootstrap=10000,seed=20260915,limits='Artifact audit, no probe refits; pooled AUC and accuracy not independently recomputed.',primary=[r for r in contrasts if r['endpoint']=='block6' and r['statement']=='combined' and r['contrast']!='delta_M_secondary'])
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=COHORTS,required=True); p.add_argument('--report-root',type=Path,default=Path('reports/2026-09-16-llama-omni2-confirmation')); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
 result=audit(args.dataset,args.report_root); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
