import csv
import numpy as np
import pytest
from scripts.analyze_module_b import validate, representations, compare_endpoints


def fixture_data():
    rows = [dict(sample_id='h', emotion='happy'), dict(sample_id='s', emotion='sad')]
    vector = np.array([[1., 2.], [-1., 2.]])
    data = dict(sample_ids=np.array(['h', 's']), projector=vector,
                audio_states=np.repeat(vector[:, None, :], 24, axis=1),
                answer_states=np.repeat(vector[:, None, :], 24, axis=1),
                answer_final_norm=vector, d_lm=np.array([1., 0.]), clean_margins=np.array([1., -1.]))
    return rows, data


def test_complete_answer_parity_and_layer_axis_validation():
    rows, data = fixture_data()
    assert validate(data, rows) == 0
    with pytest.raises(ValueError, match='order'):
        validate(data, rows[::-1])
    data['answer_states'] = data['answer_states'][:, :23]
    with pytest.raises(ValueError, match='24-layer'):
        validate(data, rows)
    rows, data = fixture_data()
    data['clean_margins'][0] = 0
    with pytest.raises(ValueError, match='parity'):
        validate(data, rows)


def test_raw_block_states_and_final_norm_are_distinct_readouts():
    _, data = fixture_data()
    items = list(representations(data))
    assert len(items) == 50
    assert items[0][:2] == ('projector', -1)
    assert items[-1][:2] == ('answer_final_norm', 24)
    assert [layer for name, layer, _ in items if name == 'answer'] == list(range(24))
    assert [layer for name, layer, _ in items if name == 'audio'] == list(range(24))


def test_module_a_reference_gate_checks_point_and_uncertainty(tmp_path):
    summary, reference = [], []
    for position, obj in [('projector', 'projector'), ('answer_final_norm', 'answer_state')]:
        for regime in ['speaker', 'statement', 'joint']:
            for metric in ['accuracy', 'roc_auc', 'mean_within_fold_auc']:
                values = dict(regime=regime, metric=metric, mean=.7, ci_low=.6 if metric != 'mean_within_fold_auc' else '', ci_high=.8 if metric != 'mean_within_fold_auc' else '')
                summary.append(dict(position=position, **values))
                reference.append(dict(object=obj, **values))
    path = tmp_path / 'reference.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference[0]))
        writer.writeheader()
        writer.writerows(reference)
    assert compare_endpoints(summary, path)['n_comparisons'] == 42
    summary[-2]['mean'] = .71
    drift = compare_endpoints(summary, path)
    assert not drift['answer_exact_match']
    assert max(abs(r['difference']) for r in drift['answer_differences']) == pytest.approx(.01)
    summary[0]['ci_low'] = .61
    with pytest.raises(ValueError, match='endpoint mismatch'):
        compare_endpoints(summary, path)


def test_probe_protocol_matches_module_a_and_oof_coverage(tmp_path):
    pytest.importorskip('sklearn')
    from scripts.analyze_module_a_baseline import analyze as analyze_a
    from scripts.analyze_module_b import analyze
    rows = [dict(sample_id=f'{a}-{s}-{e}', actor=str(a), statement=str(s), emotion=e)
            for a in range(3) for s in range(2) for e in ['happy', 'sad']]
    y = np.array([1 if r['emotion'] == 'happy' else -1 for r in rows], dtype=float)
    vector = np.stack([y, np.arange(len(y)) * .01], axis=1)
    common = dict(sample_ids=np.array([r['sample_id'] for r in rows]), projector=vector,
                  d_lm=np.array([1., 0.]), clean_margins=y)
    a_source, b_source = tmp_path / 'a.npz', tmp_path / 'b.npz'
    np.savez(a_source, **common, answer_state=vector)
    np.savez(b_source, **common, answer_final_norm=vector,
             audio_states=np.repeat(vector[:, None, :], 24, axis=1),
             answer_states=np.repeat(vector[:, None, :], 24, axis=1))
    metadata = tmp_path / 'metadata.csv'
    with metadata.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    a_out, b_out = tmp_path / 'a', tmp_path / 'b'
    analyze_a(a_source, metadata, a_out, bootstrap=30)
    summary = analyze(b_source, metadata, b_out, a_out / 'summary.csv', bootstrap=30, make_plot=False)
    assert len(summary) == 450
    assert (a_out / 'fold_membership.csv').read_bytes() == (b_out / 'fold_membership.csv').read_bytes()
    predictions = list(csv.DictReader((b_out / 'oof_predictions.csv').open()))
    keys = {(r['position'], r['layer'], r['regime'], r['sample_id']) for r in predictions}
    assert len(predictions) == len(keys) == 50 * 3 * len(rows)
