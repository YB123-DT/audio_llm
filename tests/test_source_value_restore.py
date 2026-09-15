"""Source partitions are exclusive; V hooks cannot overwrite other sources."""
import pytest
from scripts.run_source_value_restore import source_partitions, source_value_restore, _spans


def test_source_partition_and_marker_roles():
    groups = source_partitions(dict(audio_start=29, audio_length=300, prefix_length=331,
                                    decision_position=330))
    assert groups['audio'] == list(range(29, 329))
    assert groups['prompt'] == list(range(28))
    assert groups['other'] == [28, 329, 330]
    assert groups['non_audio'] == list(range(29)) + [329, 330]
    assert groups['decision'] == [330]
    assert groups['post_audio_marker'] == [329]
    assert sorted(groups['prompt']+groups['other']+groups['audio']) == groups['all']
    assert _spans(groups['other']) == [[28,29], [329,331]]


def test_partition_derives_from_layout_and_rejects_unknown_layout():
    positions = dict(audio_start=4, audio_length=5, prefix_length=11, decision_position=10)
    assert source_partitions(positions)['other'] == [3,9,10]
    with pytest.raises(ValueError, match='layout'):
        source_partitions(dict(positions, prefix_length=12))
    with pytest.raises(ValueError, match='layout'):
        source_partitions(dict(positions, decision_position=9))


def test_restore_only_source_rows_and_preserve_auxiliary():
    torch = pytest.importorskip('torch')
    hidden = torch.arange(40).reshape(2,5,4).float()
    original = hidden.clone()
    clean = hidden + 100
    auxiliary = object()
    result, aux = source_value_restore(clean, [1,3])(None, None, (hidden, auxiliary))
    assert aux is auxiliary
    assert torch.equal(hidden, original)
    assert torch.equal(result[:, [1,3]], clean[:, [1,3]])
    assert torch.equal(result[:, [0,2,4]], hidden[:, [0,2,4]])
    assert torch.equal(source_value_restore(hidden, [1,3])(None, None, hidden), hidden)
    assert torch.equal(source_value_restore(clean, list(range(5)))(None, None, hidden), clean)
    with pytest.raises(ValueError, match='identical prefix shapes'):
        source_value_restore(clean[:, :4], [1])(None, None, hidden)


def test_source_restore_changes_all_queries_that_attend_selected_source():
    torch = pytest.importorskip('torch')
    # Both query rows use source 1: source restoration is not a decision-edge hook.
    weights = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]]])
    values = torch.zeros(1,3,2)
    clean = values.clone()
    clean[:,1] = 2
    restored = source_value_restore(clean, [1])(None, None, values)
    outputs = weights @ restored
    assert torch.equal(outputs, torch.tensor([[[2.0,2.0], [1.0,1.0]]]))
