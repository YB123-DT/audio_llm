import numpy as np
import pytest

from scripts.analyze_model_replication import (
    CONDITIONS, bootstrap_counts, summarize, validate_manifest, validate_states, within_folds,
)
from scripts.analyze_answer_content import within_folds as original_folds
from scripts.analyze_crema_confirmation import within_folds as crema_folds


def manifest(n_actors=4, n_statements=2, n_repetitions=2):
    return [dict(sample_id=f'{a}-{s}-{r}-{e}', actor=f'{a:02}', statement=f'{s:02}',
                 repetition=f'{r:02}', intensity='01', emotion=e)
            for a in range(n_actors) for s in range(n_statements)
            for r in range(n_repetitions) for e in ('happy', 'sad')]


@pytest.mark.parametrize('actors,statements,repetitions,n_train,n_test', [(24, 2, 2, 92, 4), (88, 11, 1, 174, 2)])
def test_dynamic_folds_match_frozen_protocols(actors, statements, repetitions, n_train, n_test):
    rows = manifest(actors, statements, repetitions)
    folds = list(within_folds(rows))
    reference = list(original_folds(rows) if repetitions == 2 else crema_folds(rows))
    assert len(folds) == actors * statements
    coverage = []
    for (name, train, test), (old_name, old_train, old_test) in zip(folds, reference):
        assert name == old_name
        np.testing.assert_array_equal(train, old_train)
        np.testing.assert_array_equal(test, old_test)
        assert len(train) == n_train and len(test) == n_test
        assert not {rows[i]['actor'] for i in train} & {rows[i]['actor'] for i in test}
        assert len({rows[i]['statement'] for i in np.r_[train, test]}) == 1
        coverage.extend(test)
    assert sorted(coverage) == list(range(len(rows)))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'label', 'intensity', 'repetition'])
def test_reject_incomplete_or_duplicate_cells(mutation):
    rows = manifest()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(dict(rows[0], sample_id='new-id'))
    elif mutation == 'label':
        rows[0]['emotion'] = 'angry'
    elif mutation == 'intensity':
        rows[0]['intensity'] = 'HI'
    else:
        rows[0]['repetition'] = ''
    with pytest.raises(ValueError):
        validate_manifest(rows)


def test_repeat_fold_auc_and_paired_contrasts():
    rows = manifest()
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    actors = np.array([r['actor'] for r in rows])
    statements = np.array([r['statement'] for r in rows])
    # Happy [0, 2], sad [1, 3] yields AUC=.25 within each four-sample fold.
    scores = np.array([2 * int(r['repetition']) + (r['emotion'] == 'sad') for r in rows], dtype=float)
    counts = bootstrap_counts(4, 200)
    summary, contrasts = summarize(y, dict(clean=scores, no_attn=-scores, no_mlp=scores), actors, statements, counts)
    clean = next(r for r in summary if r['condition'] == 'clean' and r['metric'] == 'mean_within_fold_auc')
    assert clean['mean'] == clean['ci_low'] == clean['ci_high'] == .25
    for name in ('delta_A', 'delta_A_minus_M'):
        row = next(r for r in contrasts if r['contrast'] == name and r['metric'] == 'mean_within_fold_auc')
        assert row['mean'] == row['ci_low'] == row['ci_high'] == .5


def test_endpoint_validation():
    rows = manifest()
    data = dict(sample_ids=np.array([r['sample_id'] for r in rows]), conditions=np.array(CONDITIONS),
                block6=np.zeros((len(rows), 3, 4)), final_answer=np.zeros((len(rows), 3, 4)))
    validate_states(data, rows)
    data['block6'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid endpoint'):
        validate_states(data, rows)
