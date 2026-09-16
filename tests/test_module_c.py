import csv
import numpy as np
import pytest
from scripts import analyze_module_c as module
from scripts.analyze_answer_content import within_folds
from scripts.analyze_edge_representation import folds


def rows():
    return [dict(sample_id=f'{a}-{s}-{r}-{e}', actor=f'{a:02}', statement=f'{s:02}', emotion=e) for a in range(1, 25) for s in (1, 2) for r in (1, 2) for e in ('happy', 'sad')]


def test_paired_delta_not_interval_subtraction():
    a = (2., np.array([1., 2., 3.]))
    b = (1., np.array([0., 1., 2.]))
    assert module.contrast(a, b) == dict(mean=1., ci_low=1., ci_high=1.)


def test_fold_test_identity_and_train_isolation():
    metadata = rows()
    for (name, train, test), (other, cross, ctest) in zip(within_folds(metadata), folds(metadata, 'joint')):
        assert name == other and np.array_equal(test, ctest)
        assert len(train) == len(cross) == 92 and len(test) == 4
        actor, statement = metadata[test[0]]['actor'], metadata[test[0]]['statement']
        assert all(metadata[i]['actor'] != actor and metadata[i]['statement'] == statement for i in train)
        assert all(metadata[i]['actor'] != actor and metadata[i]['statement'] != statement for i in cross)


def fixture_states():
    states = np.arange(2 * 24 * 3).reshape(2, 24, 3).astype(float)
    reference = dict(answer_states=states, answer_final_norm=np.ones((2, 3)), clean_margins=np.zeros(2), d_lm=np.ones(3))
    data = dict(sample_ids=np.array(['a', 'b']), answer_in=np.concatenate([np.zeros((2, 1, 3)), states[:, :-1]], axis=1), answer_attn=states.copy(), answer_out=states.copy(), **{k: v.copy() for k, v in reference.items() if k != 'answer_states'})
    return data, reference, [dict(sample_id='a'), dict(sample_id='b')]


@pytest.mark.parametrize('corruption', ['none', 'identity', 'output', 'continuity', 'nan'])
def test_exact_vector_reuse_gate(monkeypatch, corruption):
    monkeypatch.setattr(module, 'validate_reference', lambda *args: None)
    data, reference, metadata = fixture_states()
    if corruption == 'identity':
        data['sample_ids'] = data['sample_ids'][::-1]
    elif corruption == 'output':
        data['answer_out'][0, 0, 0] += 1e-10
    elif corruption == 'continuity':
        data['answer_in'][0, 1, 0] += 1e-10
    elif corruption == 'nan':
        data['answer_attn'][0, 0, 0] = np.nan
    if corruption == 'none':
        assert all(v == 0 for v in module.validate_states(data, reference, metadata).values())
    else:
        with pytest.raises(ValueError):
            module.validate_states(data, reference, metadata)


@pytest.mark.parametrize('corruption', ['none', 'duplicate', 'missing', 'fold', 'metadata', 'score'])
def test_cache_gate(tmp_path, corruption):
    metadata = rows()
    records = [dict(position='answer', layer=l, condition=c, fold=f"actor_{r['actor']}_statement_{r['statement']}", **r, score=0., prediction='sad') for l in range(24) for c in ('within', 'cross') for r in metadata]
    if corruption == 'duplicate':
        records.append(records[0])
    elif corruption == 'missing':
        records.pop()
    elif corruption == 'fold':
        records[0]['fold'] = 'bad'
    elif corruption == 'metadata':
        records[0]['actor'] = '99'
    elif corruption == 'score':
        records[0]['score'] = float('nan')
    path = tmp_path / 'cache.csv'
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if corruption == 'none':
        assert len(module.read_cached(path, metadata)) == 24 * 2 * 192
    else:
        with pytest.raises(ValueError):
            module.read_cached(path, metadata)


def test_constant_balanced_probe_chance():
    sklearn = pytest.importorskip('sklearn')
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    x, y = np.ones((96, 3)), np.tile([1, 0], 48)
    scaler = StandardScaler().fit(x[:92])
    clf = LogisticRegression(C=1., max_iter=5000, random_state=module.SEED).fit(scaler.transform(x[:92]), y[:92])
    values = clf.decision_function(scaler.transform(x[92:]))
    assert np.array_equal(values, np.zeros(4))
    assert module.auc(y[92:], values) == .5


def test_telescoping_changes_preserve_pairing():
    rng = np.random.default_rng(42)
    initial, attn, out = rng.normal(size=(3, 23, 10))
    initial[1:] = out[:-1]
    delta_a, delta_m = attn - initial, out - attn
    assert np.allclose((delta_a + delta_m).sum(axis=0), out[-1] - initial[0])
