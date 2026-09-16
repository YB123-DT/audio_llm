import csv
import numpy as np
import pytest
from scripts.analyze_answer_content import within_folds, read_cross, paired_summary
from scripts.analyze_edge_representation import folds


def rows():
    return [dict(sample_id=f'{a}-{s}-{r}-{e}', actor=f'{a:02}', statement=f'{s:02}', emotion=e) for a in range(1, 25) for s in (1, 2) for r in (1, 2) for e in ('happy', 'sad')]


def test_within_cross_isolation_and_size():
    metadata = rows()
    cross = list(folds(metadata, 'joint'))
    for (name, train, test), (other, ctrain, ctest) in zip(within_folds(metadata), cross):
        assert name == other and np.array_equal(test, ctest)
        assert len(train) == len(ctrain) == 92 and len(test) == 4
        actor, statement = metadata[test[0]]['actor'], metadata[test[0]]['statement']
        assert all(metadata[i]['actor'] != actor and metadata[i]['statement'] == statement for i in train)
        assert all(metadata[i]['actor'] != actor and metadata[i]['statement'] != statement for i in ctrain)


def test_paired_bootstrap_preserves_zero_gap():
    y = np.array([1, 0, 1, 0])
    scores = np.array([2., 1., 0., 1.])
    actors = np.array(['01', '01', '02', '02'])
    statements = np.array(['01'] * 4)
    result = paired_summary(y, scores, scores, actors, statements, np.array([[2, 0], [1, 1], [0, 2]]))
    assert all(r['mean'] == r['ci_low'] == r['ci_high'] == 0 for r in result if r['condition'] == 'gap')
    result = paired_summary(y, scores, -scores, actors, statements, np.array([[2, 0], [0, 2]]))
    gap = next(r for r in result if r['condition'] == 'gap' and r['metric'] == 'mean_within_fold_auc')
    assert gap['mean'] == 0 and gap['ci_low'] == pytest.approx(-.95) and gap['ci_high'] == pytest.approx(.95)


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'fold', 'metadata'])
def test_cross_cache_rejects_invalid_records(tmp_path, corruption):
    metadata = rows()
    records = [dict(position='answer', layer=0, regime='joint', fold=f"actor_{r['actor']}_statement_{r['statement']}", **r, score=1, prediction='happy') for r in metadata]
    if corruption == 'duplicate':
        records.append(records[0])
    elif corruption == 'missing':
        records.pop()
    elif corruption == 'fold':
        records[0]['fold'] = 'wrong'
    else:
        records[0]['actor'] = '99'
    path = tmp_path / 'cross.csv'
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError):
        read_cross(path, metadata, [('answer', 0)])
