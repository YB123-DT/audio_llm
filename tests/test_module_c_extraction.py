import numpy as np
import pytest
from scripts.extract_module_c_states import aligned_reference, install_observers, parity_checks


def test_reference_reorders_and_rejects_ambiguous_ids():
    ref = dict(sample_ids=np.array(['b', 'a']), answer_states=np.array([2, 1]), answer_final_norm=np.array([4, 3]), clean_margins=np.array([6, 5]))
    assert aligned_reference(ref, ['a', 'b'])['answer_states'].tolist() == [1, 2]
    for ids in (['a', 'c'], ['a', 'a']):
        with pytest.raises(ValueError):
            aligned_reference(ref, ids)


def test_real_qwen_hooks_capture_residuals_without_changing_forward():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    torch.manual_seed(1234)
    model = Qwen2Model(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval()
    inputs = torch.randn(1, 6, 16)
    with torch.inference_mode():
        expected = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        captured = {}
        handles = install_observers(model.layers, 5, captured)
        try:
            actual = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        finally:
            for h in handles:
                h.remove()
    assert torch.equal(expected, actual)
    assert len(captured) == 2
    for c in captured.values():
        np.testing.assert_array_equal(c['in'] + c['attention_update'], c['attn'])
        np.testing.assert_array_equal(c['attn'] + c['mlp_update'], c['out'])
    np.testing.assert_array_equal(captured[0]['in'], inputs[0, 5].numpy())
    np.testing.assert_array_equal(captured[1]['in'], captured[0]['out'])


def test_parity_detects_interior_state_corruption():
    out = np.arange(12.).reshape(2, 3, 2)
    inp = np.concatenate([np.zeros((2, 1, 2)), out[:, :-1]], axis=1)
    arrays = dict(answer_in=inp, answer_out=out.copy(), answer_final_norm=out[:, -1], clean_margins=np.zeros(2))
    reference = dict(answer_states=out, answer_final_norm=out[:, -1], clean_margins=np.zeros(2))
    assert all(v == 0 for v in parity_checks(arrays, reference).values())
    arrays['answer_out'][0, 1, 0] += 1
    assert parity_checks(arrays, reference)['module_b_all_block_outputs_max'] == 1
    assert parity_checks(arrays, reference)['adjacent_block_residual_max'] == 1
