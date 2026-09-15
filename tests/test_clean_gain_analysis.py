import csv,json
import numpy as np
import pytest
from scripts.analyze_clean_edge_gain import auc_samples,analyze,ALPHAS


def test_auc_cluster_recomputation_uses_cross_actor_pairs():
    x=np.array([[[2.,1.]],[[0.,3.]]])
    mean,boots=auc_samples(x,np.array([[1,1],[2,0],[0,2]]))
    assert mean==.25
    assert boots.tolist()==[.25,1.,0.]
    assert auc_samples(np.ones_like(x),np.array([[1,1]]))[0]==.5


def test_uniform_gain_shift_is_not_emotion_signal(tmp_path):
    rows=[];ref=[]
    for a in ['01','02']:
        for emotion,val in [('happy',-1.),('sad',-2.)]:
            meta=dict(sample_id=a+emotion,pair_id=a,actor=a,statement='01',repetition='01',emotion=emotion)
            ref.append(dict(meta,condition='edge_decision_joint',clean_margin=val))
            for alpha in ALPHAS:rows.append(dict(meta,alpha=alpha,clean_margin=val,margin=val+10*(alpha-1)))
    analyze(rows,tmp_path,ref)
    with (tmp_path/'necessity_summary.csv').open() as f:n={r['metric']:r for r in csv.DictReader(f)}
    assert float(n['delta_C']['mean'])==0
    assert float(n['C_happy']['mean'])==10
    with (tmp_path/'gain_summary.csv').open() as f:rs=list(csv.DictReader(f))
    assert all(float(r['mean'])==1 for r in rs if r['metric']=='roc_auc')
    with pytest.raises(ValueError):analyze(rows+rows[:1],tmp_path,ref)
