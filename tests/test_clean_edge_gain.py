"""Natural gain intervention preserves routing and scales only the selected edge."""
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.run_clean_edge_gain import gain_output, ALPHAS
from scripts.run_audio_edge_restore import selected_attention


def test_declared_gain_grid():
    assert ALPHAS == (0., .5, 1., 1.5, 2.)


@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'additive'])
def test_gain_matches_explicit_value_scaling_with_gqa(mask_kind):
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(71)
    q = torch.randn(1, 4, 7, 3, generator=generator)
    k = torch.randn(1, 2, 7, 3, generator=generator).repeat_interleave(2, dim=1)
    v = torch.randn(1, 2, 7, 3, generator=generator).repeat_interleave(2, dim=1)
    allowed = torch.ones(7, 7, dtype=torch.bool).tril()
    mask = None if mask_kind == 'none' else allowed[None, None]
    if mask_kind == 'additive':
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float('-inf'))
    sources, decision, scale = [2, 3, 4], 6, 3**-.5
    current = selected_attention(torch, q, k, v, list(range(7)), mask, scale)
    saved = v.clone()
    for alpha in ALPHAS:
        actual = gain_output(torch, current, q, k, v, sources, decision, mask, scale, alpha)
        modified_v = v.clone()
        modified_v[:, :, sources] *= alpha
        expected = selected_attention(torch, q, k, modified_v, [decision], mask, scale)
        torch.testing.assert_close(actual[:, decision:decision+1], expected, atol=1e-6, rtol=1e-5)
        assert torch.equal(actual[:, :decision], current[:, :decision])
        if alpha == 1.:
            assert torch.equal(actual, current)
    assert torch.equal(v, saved)


def test_gain_does_not_renormalize_attention():
    torch = pytest.importorskip('torch')
    q = torch.tensor([[[[0.], [4.], [4.]]]])
    k = torch.tensor([[[[1.], [-1.], [0.]]]])
    v = torch.tensor([[[[2.], [10.], [5.]]]])
    current = selected_attention(torch, q, k, v, [0, 1, 2], None, 1.)
    actual = gain_output(torch, current, q, k, v, [0, 1], 2, None, 1., 0.)
    weights = torch.softmax(torch.tensor([4., -4., 0.]), dim=0)
    torch.testing.assert_close(actual[0, 2, 0], weights[2]*5, atol=1e-6, rtol=1e-5)
    assert abs(float(actual[0, 2, 0])-5.) > 1.
    assert torch.equal(actual[:, :2], current[:, :2])


def test_future_source_is_causally_invisible():
    torch = pytest.importorskip('torch')
    q = k = torch.ones(1, 1, 4, 1)
    v = torch.arange(4.).view(1, 1, 4, 1)
    current = selected_attention(torch, q, k, v, list(range(4)), None, 1.)
    actual = gain_output(torch, current, q, k, v, [2, 3], 1, None, 1., 2.)
    assert torch.equal(actual, current)


def test_nonfinite_gain_rejected():
    with pytest.raises(ValueError, match='Finite'):
        gain_output(None, None, None, None, None, [], 0, None, 1., float('nan'))


def test_direct_script_entrypoint_without_repository_on_pythonpath(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/run_clean_edge_gain.py'
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--manifest' in result.stdout
