import numpy as np
from scripts.analyze_edge_representation import folds, raw_direction, cosine, clustered_metrics


def test_heldout_splits_have_no_leakage_and_exact_coverage():
    rows = [dict(actor=str(a), statement=str(s), emotion=e) for a in range(3) for s in range(2) for e in ['happy', 'sad']]
    for regime, expected in [('speaker', 3), ('statement', 2), ('joint', 6)]:
        split = list(folds(rows, regime))
        assert len(split) == expected
        coverage = np.zeros(len(rows), dtype=int)
        for _, train, test in split:
            coverage[test] += 1
            assert not set(train) & set(test)
            for field in (['actor', 'statement'] if regime == 'joint' else ['actor' if regime == 'speaker' else 'statement']):
                assert not {rows[i][field] for i in train} & {rows[i][field] for i in test}
        assert np.all(coverage == 1)


def test_scaled_probe_direction_is_converted_to_residual_coordinates():
    coef = np.array([2., 3.])
    scale = np.array([2., 6.])
    mean = np.array([8., -4.])
    x = np.array([3., 7.])
    raw = raw_direction(coef, scale)
    assert np.isclose(np.dot((x - mean) / scale, coef), np.dot(x, raw) - np.dot(mean, raw))
    assert np.isclose(cosine(raw, raw), 1.)
    assert not np.isclose(cosine(raw, coef), 1.)


def test_actor_bootstrap_auc_uses_cross_actor_comparisons():
    y = np.array([1, 0, 1, 0])
    score = np.array([2., 1., 0., -1.])
    actor = np.array(['a', 'a', 'b', 'b'])
    counts = np.array([[2, 0], [0, 2], [1, 1]])
    result = {r['metric']: r for r in clustered_metrics(y, score, actor, counts)}
    assert result['roc_auc']['mean'] == .75
    assert result['roc_auc']['ci_high'] == 1.
    assert result['accuracy']['mean'] == .5


def test_end_to_end_probe_fits_train_only_and_emits_all_outputs(tmp_path):
    import pytest
    pytest.importorskip('sklearn')
    import csv
    from scripts.analyze_edge_representation import analyze
    rows = [dict(sample_id=f'{a}_{s}_{e}', actor=str(a), statement=str(s), emotion=e) for a in range(3) for s in range(2) for e in ['happy', 'sad']]
    labels = np.array([1. if r['emotion'] == 'happy' else -1. for r in rows])
    vectors = np.stack([labels, np.arange(len(rows)) * .01], axis=1)[:, None, :]
    path = tmp_path / 'vectors.npz'
    np.savez(path, edge_updates=vectors, layers=np.array([18]), sample_ids=np.array([r['sample_id'] for r in rows]), d_lm=np.array([1., 0.]), clean_margins=labels)
    metadata = tmp_path / 'metadata.csv'
    with metadata.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / 'out'
    analyze(path, metadata, out, bootstrap=20)
    predictions = list(csv.DictReader((out / 'oof_predictions.csv').open()))
    assert len(predictions) == 3 * len(rows)
    assert all(r['prediction'] == r['emotion'] for r in predictions)
    metrics = list(csv.DictReader((out / 'fold_metrics.csv').open()))
    assert len(metrics) == 11
    assert max(float(r['raw_score_parity_max']) for r in metrics) < 1e-8
    assert (out / 'edge_probes.png').exists()
