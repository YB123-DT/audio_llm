import numpy as np
import pytest
from scripts.analyze_edge_content_counterfactual import calibration_indices, project, auc


def test_calibration_excludes_entire_held_actor():
    rows = [dict(actor=a, statement=s) for a in ['01', '02', '03'] for s in ['01', '02'] for _ in range(2)]
    source, target = calibration_indices(rows, np.array([2, 3]))
    assert all(rows[i]['actor'] != '01' for i in np.r_[source, target])
    assert all(rows[i]['statement'] == '01' for i in source)
    assert all(rows[i]['statement'] == '02' for i in target)


def test_rank1_projection_removes_only_axis():
    x = np.array([[1., 2.], [-1., 3.]])
    np.testing.assert_allclose(project(x, np.array([2., 0.])), [[0., 2.], [0., 3.]])
    np.testing.assert_array_equal(project(x, np.zeros(2)), x)


def test_offset_recovery_and_rotation_failure():
    pytest.importorskip('sklearn')
    from scripts.analyze_edge_content_counterfactual import fit_probe
    y = np.tile([0, 1], 20)
    source = np.column_stack([2*y-1, np.tile([-.1, -.1, .1, .1], 10)])
    target = source + [8., 0.]
    w, b = fit_probe(source, y)
    raw = target @ w + b
    fixed = raw - w @ (target.mean(0)-source.mean(0))
    assert ((raw > 0) == y).mean() == .5
    assert ((fixed > 0) == y).mean() == 1
    assert auc(y, raw) == auc(y, fixed)
    rotated = -source
    corrected = rotated @ w + b - w @ (rotated.mean(0)-source.mean(0))
    assert ((corrected > 0) == y).mean() == 0


def test_projected_probe_training_does_not_use_test_values():
    pytest.importorskip('sklearn')
    from scripts.analyze_edge_content_counterfactual import fit_probe
    source = np.array([[2., -1.], [2., 1.], [3., -2.], [3., 2.]])
    offset = np.array([5., 0.])
    w1, b1 = fit_probe(project(source, offset), np.array([0, 1, 0, 1]))
    # Applying the learned projection to arbitrary test values cannot alter its fit.
    project(np.full((7, 2), 1e9), offset)
    w2, b2 = fit_probe(project(source, offset), np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(w1, w2)
    assert b1 == b2
