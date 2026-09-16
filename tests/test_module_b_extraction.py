import numpy as np
import pytest
from scripts.extract_module_b_states import aligned_reference, pooled_states


def test_reference_requires_identity_and_aligns_order():
    data = dict(sample_ids=np.array(['b', 'a']), projector=np.array([[2], [1]]), answer_state=np.array([[4], [3]]), clean_margins=np.array([6, 5]))
    got = aligned_reference(data, ['a', 'b'])
    assert got['projector'].ravel().tolist() == [1, 2]
    with pytest.raises(ValueError):
        aligned_reference(data, ['a', 'c'])
    data['sample_ids'] = np.array(['a', 'a'])
    with pytest.raises(ValueError):
        aligned_reference(data, ['a', 'b'])


def test_complete_answer_and_audio_span_are_separate():
    torch = pytest.importorskip('torch')
    h = torch.arange(30.).reshape(1, 6, 5)
    audio, answer = pooled_states(h, 1, 3, 5)
    assert torch.equal(audio, h[0, 1:4].mean(0))
    assert torch.equal(answer, h[0, 5])
    with pytest.raises(ValueError):
        pooled_states(h, 1, 5, 5)
