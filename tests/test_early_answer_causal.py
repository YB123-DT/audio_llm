import json

import numpy as np
import pytest

from scripts.analyze_early_answer_causal import (
    CONDITIONS, fit_probe, sha, summarize, validate_cache_provenance, validate_states,
)
from scripts.analyze_answer_content import within_folds


def test_within_actor_isolation_and_same_statement():
    rows = [dict(sample_id=f'{a}-{s}-{r}-{e}', actor=f'{a:02}', statement=f'{s:02}', emotion=e)
            for a in range(1, 25) for s in (1, 2) for r in (1, 2) for e in ('happy', 'sad')]
    splits = list(within_folds(rows))
    assert len(splits) == 48
    for _, train, test in splits:
        assert len(train) == 92 and len(test) == 4
        actor, statement = rows[test[0]]['actor'], rows[test[0]]['statement']
        assert all(rows[i]['actor'] != actor and rows[i]['statement'] == statement for i in train)


def test_direct_paired_contrasts_preserve_correlated_uncertainty():
    y = np.array([1, 0, 1, 0])
    actors = np.array(['01', '01', '02', '02'])
    statements = np.array(['01'] * 4)
    scores = np.array([2., 1., 0., 1.])
    counts = np.array([[2, 0], [0, 2]])
    _, contrasts = summarize(y, dict(clean=scores, no_attn=scores, no_mlp=-scores), actors, statements, counts)
    a = next(r for r in contrasts if r['contrast'] == 'delta_A' and r['metric'] == 'mean_within_fold_auc')
    assert a['mean'] == a['ci_low'] == a['ci_high'] == 0
    am = next(r for r in contrasts if r['contrast'] == 'delta_A_minus_M' and r['metric'] == 'mean_within_fold_auc')
    assert am['mean'] == 0
    assert am['ci_low'] == pytest.approx(-.95)
    assert am['ci_high'] == pytest.approx(.95)


def test_exact_clean_vector_gate():
    rows = [dict(sample_id='a'), dict(sample_id='b')]
    clean = dict(sample_ids=np.array(['a', 'b']), answer_states=np.zeros((2, 24, 3)), answer_final_norm=np.ones((2, 3)))
    data = dict(sample_ids=clean['sample_ids'], conditions=np.array(CONDITIONS), block6=np.zeros((2, 3, 3)), final_answer=np.ones((2, 3, 3)))
    validate_states(data, clean, rows)
    data['block6'][0, 0, 0] = 1e-12
    with pytest.raises(ValueError, match='Clean vectors'):
        validate_states(data, clean, rows)
    data['block6'][0, 0, 0] = 0
    data['sample_ids'] = np.array(['b', 'a'])
    with pytest.raises(ValueError, match='identity'):
        validate_states(data, clean, rows)


def test_cache_hash_gate(tmp_path):
    from pathlib import Path
    import scripts.analyze_early_answer_causal as module
    source, metadata = tmp_path / 'states.npz', tmp_path / 'metadata.csv'
    source.write_bytes(b'original states')
    metadata.write_text('original metadata')
    analyzer = Path(module.__file__).with_name('analyze_answer_content.py')
    analysis = tmp_path / 'analysis.json'
    info = dict(sources_sha256={str(p): sha(p) for p in (source, metadata, analyzer)}, seed=20260915, n_samples=192, n_actors=24)
    analysis.write_text(json.dumps(info))
    validate_cache_provenance(analysis, source, metadata)
    metadata.write_text('changed metadata')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_cache_provenance(analysis, source, metadata)


def test_probe_scaler_is_fit_on_training_only():
    pytest.importorskip('sklearn')
    x = np.array([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.], [0., 0.], [1e9, -1e9]])
    y = np.array([0, 0, 1, 1, 0, 1])
    train = np.arange(4)
    before = fit_probe(x, y, train, np.array([4]))
    after = fit_probe(x, y, train, np.array([4, 5]))
    assert np.array_equal(before, after[:1])
    with pytest.raises(ValueError, match='overlap'):
        fit_probe(x, y, train, np.array([3]))
