import numpy as np
import pytest

from scripts.analyze_crema_confirmation import (
    CONDITIONS, bootstrap_counts, summarize, validate_manifest, validate_states, within_folds,
)


def manifest():
    return [dict(sample_id=f'{a}-{s}-{e}', actor=str(a), statement=s, intensity='XX', emotion=e)
            for a in range(4) for s in ('AAA', 'BBB', 'CCC') for e in ('happy', 'sad')]


def test_dynamic_folds_are_actor_isolated_and_content_matched():
    rows = manifest()
    seen = []
    splits = list(within_folds(rows))
    assert len(splits) == 12
    for _, train, test in splits:
        assert len(train) == 6 and len(test) == 2
        assert not set(train) & set(test)
        actor, statement = rows[test[0]]['actor'], rows[test[0]]['statement']
        assert all(rows[i]['actor'] != actor and rows[i]['statement'] == statement for i in train)
        assert {rows[i]['emotion'] for i in test} == {'happy', 'sad'}
        seen.extend(test)
    assert sorted(seen) == list(range(len(rows)))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'label', 'intensity'])
def test_invalid_rectangular_coverage_is_rejected(mutation):
    rows = manifest()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(dict(rows[0], sample_id='new-id'))
    elif mutation == 'label':
        rows[0]['emotion'] = 'angry'
    else:
        rows[0]['intensity'] = 'HI'
    with pytest.raises(ValueError):
        validate_manifest(rows)


def test_states_require_identity_and_finite_all_conditions():
    rows = manifest()
    data = dict(sample_ids=np.array([r['sample_id'] for r in rows]), conditions=np.array(CONDITIONS), block6=np.zeros((24, 3, 4)), final_answer=np.zeros((24, 3, 4)))
    validate_states(data, rows)
    data['block6'][0, 2, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid endpoint'):
        validate_states(data, rows)
    data['block6'][0, 2, 0] = 0
    data['sample_ids'] = data['sample_ids'][::-1]
    with pytest.raises(ValueError, match='identity'):
        validate_states(data, rows)


def test_shared_actor_bootstrap_uses_direct_paired_contrasts():
    rows = manifest()
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    scores = np.array([float(y[i]) if int(r['actor']) < 2 else -float(y[i]) for i, r in enumerate(rows)])
    counts = bootstrap_counts(4, 200)
    assert np.array_equal(counts, bootstrap_counts(4, 200))
    assert np.all(counts.sum(axis=1) == 4)
    summary, contrasts = summarize(y, dict(clean=scores, no_attn=scores, no_mlp=-scores), actors, statements, counts)
    a = next(r for r in contrasts if r['contrast'] == 'delta_A' and r['metric'] == 'mean_within_fold_auc')
    assert a['mean'] == a['ci_low'] == a['ci_high'] == 0
    am = next(r for r in contrasts if r['contrast'] == 'delta_A_minus_M' and r['metric'] == 'mean_within_fold_auc')
    assert am['mean'] == 0
    assert am['ci_low'] < 0 < am['ci_high']
    clean = next(r for r in summary if r['condition'] == 'clean' and r['metric'] == 'mean_within_fold_auc')
    assert clean['mean'] == .5
    with pytest.raises(ValueError, match='Positive'):
        bootstrap_counts(4, 0)
