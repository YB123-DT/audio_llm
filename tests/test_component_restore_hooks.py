"""Contribution hook invariants; run on remote runtime when torch is available."""
import pytest

torch = pytest.importorskip('torch')
from scripts.run_residual_component_restore import contribution_restore, condition_specs


def test_decision_restore_preserves_other_positions_and_tuple_auxiliary():
    hidden = torch.arange(24).reshape(2, 3, 4).float()
    original = hidden.clone()
    clean = torch.full((2, 4), -3.0)
    auxiliary = object()
    result, aux = contribution_restore(clean, 1)(None, None, (hidden, auxiliary))
    assert aux is auxiliary
    assert torch.equal(hidden, original)
    assert torch.equal(result[:, 0], hidden[:, 0])
    assert torch.equal(result[:, 2], hidden[:, 2])
    assert torch.equal(result[:, 1], clean)
    # The hook returns a contribution; the caller's residual stays additive.
    residual = torch.ones_like(result) * 10
    assert torch.equal((result + residual)[:, 1], clean + 10)


def test_all_source_restore_and_shape_guard():
    hidden = torch.zeros(1, 3, 4)
    clean = torch.ones_like(hidden)
    result = contribution_restore(clean, 2, all_positions=True)(None, None, hidden)
    assert torch.equal(result, clean)
    assert not torch.equal(hidden, clean)
    with pytest.raises(ValueError, match='identical prefix shapes'):
        contribution_restore(clean[:, :2], 2, all_positions=True)(None, None, hidden)


def test_noop_and_qkv_conditional_schedules():
    hidden = torch.randn(1, 3, 4)
    assert torch.equal(contribution_restore(hidden[:, 2], 2)(None, None, hidden), hidden)
    defaults = condition_specs([])
    assert len(defaults) == 14
    assert all(kind in ('attention', 'mlp') for _, kind, _ in defaults)
    qkv = condition_specs([18, 19])
    assert ('qk_joint', 'qk', [18, 19]) in qkv
    assert ('v_joint', 'v', [18, 19]) in qkv
    assert ('qkv_joint', 'qkv', [18, 19]) in qkv
