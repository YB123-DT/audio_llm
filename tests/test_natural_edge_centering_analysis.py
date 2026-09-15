import csv
import numpy as np
import pytest
from scripts.analyze_natural_edge_centering import analyze, CONDITIONS, auc_samples


def test_cluster_auc_respects_actor_multiplicities_and_ties():
    values = np.array([[[2., 1.]], [[0., 3.]]])
    mean, boots = auc_samples(values, np.array([[1, 1], [2, 0], [0, 2]]))
    assert mean == .25
    assert boots.tolist() == [.25, 1., 0.]
    assert auc_samples(np.ones_like(values), np.array([[1, 1]]))[0] == .5


def fixture_rows():
    rows, reference = [], []
    for actor in ['01', '02']:
        for statement in ['01', '02']:
            for emotion, baseline in [('happy', -2.), ('sad', -1.)]:
                meta = dict(sample_id=actor+statement+emotion, actor=actor, statement=statement,
                            pair_id=actor+statement, repetition='01', emotion=emotion)
                reference.append(dict(meta, condition='edge_decision_joint', clean_margin=baseline))
                for condition in CONDITIONS:
                    shift = 3. if condition == 'center_joint' else 0.
                    if condition == 'center_19': shift = 2. if emotion == 'happy' else -2.
                    rows.append(dict(meta, condition=condition, margin=baseline+shift, clean_margin=baseline))
    return rows, reference


def test_common_shift_changes_threshold_not_auc_but_differential_shift_changes_ranking(tmp_path):
    rows, reference = fixture_rows()
    summary, contrasts = analyze(rows, tmp_path, reference)
    metrics = {(r['condition'], r['metric']): r['mean'] for r in summary}
    assert metrics['clean', 'roc_auc'] == metrics['center_joint', 'roc_auc'] == 0
    assert metrics['center_joint', 'pair_separation'] == -1
    assert metrics['center_joint', 'common_shift'] == 3
    assert metrics['center_joint', 'happy_prediction_fraction'] == 1
    assert metrics['center_19', 'roc_auc'] == 1
    assert metrics['center_19', 'pair_separation'] == 3
    assert metrics['center_19', 'common_shift'] == 0
    changes = {(r['condition'], r['metric']): r['mean'] for r in contrasts}
    assert changes['center_19', 'roc_auc_minus_clean'] == 1
    assert changes['center_joint', 'roc_auc_minus_clean'] == 0
    with (tmp_path/'pair_shifts.csv').open() as f: pairs = list(csv.DictReader(f))
    assert all(float(r['delta_pair_separation']) == 4 for r in pairs if r['condition'] == 'center_19')


def test_malformed_or_nonidentical_baselines_rejected(tmp_path):
    rows, reference = fixture_rows()
    with pytest.raises(ValueError): analyze(rows+rows[:1], tmp_path, reference)
    bad = [dict(r) for r in rows]
    next(r for r in bad if r['condition'] == 'zero_noop')['margin'] += .1
    with pytest.raises(ValueError): analyze(bad, tmp_path, reference)
