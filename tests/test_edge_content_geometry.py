import numpy as np
from scripts.analyze_edge_content_geometry import geometry, raw_coefficients


def test_shared_direction_offset():
    means = np.array([[[1., 0.], [-1., 0.]], [[4., 2.], [2., 2.]]])
    stats = geometry(means)
    assert np.isclose(stats['cos_emotion_directions'], 1)
    assert np.isclose(stats['offset_to_gap_dshared'], 1.5)
    assert np.isclose(stats['norm_offset'], np.sqrt(13))


def test_rotated_direction_batched():
    means = np.array([[[1., 0.], [-1., 0.]], [[0., 1.], [0., -1.]]])
    stats = geometry(np.stack([means, means]))
    np.testing.assert_allclose(stats['cos_emotion_directions'], 0)
    np.testing.assert_allclose(stats['norm_offset'], 0)


def test_raw_coordinate_coefficients():
    x = np.array([[2., 4.], [5., 7.]])
    mean, scale = np.array([1., 3.]), np.array([2., 4.])
    coefficient, intercept = np.array([[3., -2.]]), np.array([.5])
    w, b = raw_coefficients(coefficient, intercept, mean, scale)
    np.testing.assert_allclose(x @ w + b, ((x - mean) / scale) @ coefficient[0] + intercept[0])
