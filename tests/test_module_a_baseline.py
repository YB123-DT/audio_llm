import numpy as np
import pytest
from scripts.analyze_module_a_baseline import validate, folds


def test_full_answer_head_parity_and_order_gate():
    rows=[dict(sample_id='h',emotion='happy'),dict(sample_id='s',emotion='sad')]
    data=dict(sample_ids=np.array(['h','s']),projector=np.ones((2,2)),answer_state=np.array([[1.,2.],[3.,4.]]),d_lm=np.array([1.,-1.]),clean_margins=np.array([-1.,-1.]))
    assert validate(data,rows)==0
    data['clean_margins'][0]=0
    with pytest.raises(ValueError,match='parity'): validate(data,rows)
    with pytest.raises(ValueError,match='order'): validate(data,rows[::-1])


def test_unified_folds_disjoint_complete():
    rows=[dict(actor=str(a),statement=str(s),emotion=e) for a in range(3) for s in range(2) for e in ['happy','sad']]
    for regime in ['speaker','statement','joint']:
        coverage=np.zeros(len(rows),dtype=int)
        for _,train,test in folds(rows,regime):
            assert not set(train)&set(test)
            coverage[test]+=1
            for key in (['actor'] if regime=='speaker' else ['statement'] if regime=='statement' else ['actor','statement']):
                assert not {rows[i][key] for i in train}&{rows[i][key] for i in test}
        assert np.all(coverage==1)


def test_native_scores_reused_without_fit_and_table_complete(tmp_path):
    pytest.importorskip('sklearn')
    import csv
    from scripts.analyze_module_a_baseline import analyze
    rows=[dict(sample_id=f'{a}-{s}-{e}',actor=str(a),statement=str(s),emotion=e) for a in range(3) for s in range(2) for e in ['happy','sad']]
    y=np.array([1 if r['emotion']=='happy' else -1 for r in rows],dtype=float)
    vectors=np.stack([y,np.arange(len(y))*.01],axis=1)
    source=tmp_path/'vectors.npz'
    np.savez(source,sample_ids=np.array([r['sample_id'] for r in rows]),projector=vectors,answer_state=vectors,d_lm=np.array([1.,0.]),clean_margins=y)
    metadata=tmp_path/'metadata.csv'
    with metadata.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    out=tmp_path/'out'
    analyze(source,metadata,out,bootstrap=30)
    results=list(csv.DictReader((out/'summary.csv').open()))
    assert len(results)==27
    native=[r for r in results if r['object']=='native_lm']
    assert all(float(r['mean'])==1 for r in native)
    metrics=list(csv.DictReader((out/'fold_metrics.csv').open()))
    assert all(int(r['n_train'])==0 for r in metrics if r['object']=='native_lm')
    assert all(int(r['n_train'])>0 for r in metrics if r['object']!='native_lm')


def test_legacy_projector_explicit_id_alignment():
    from scripts.analyze_module_a_baseline import align_legacy
    rows=[dict(sample_id='sad',representation_index='1'),dict(sample_id='happy',representation_index='0')]
    vectors=np.array([[1.,2.],[3.,4.]])
    assert np.array_equal(align_legacy(rows,['sad','happy'],vectors), vectors[::-1])
    with pytest.raises(ValueError,match='IDs'): align_legacy(rows,['happy','missing'],vectors)
    rows[0]['representation_index']='0'
    with pytest.raises(ValueError,match='indices'): align_legacy(rows,['happy','sad'],vectors)
