"""Structural and numerical tests of source x query value-edge restoration."""
import pytest
from scripts.run_audio_edge_restore import query_partitions, selected_attention, edge_output


def test_query_partitions():
    groups = query_partitions(dict(audio_start=3, audio_length=4, prefix_length=9, decision_position=8))
    assert groups == dict(decision=[8], audio=[3, 4, 5, 6], end_marker=[7], non_decision=list(range(8)), all=list(range(9)))
    assert set(groups['decision']).isdisjoint(groups['non_decision'])
    assert groups['decision'] + groups['non_decision'] != groups['all']
    assert sorted(groups['decision'] + groups['non_decision']) == groups['all']


@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'additive'])
def test_gqa_edge_restore_equivalence_and_query_isolation(mask_kind):
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(1, 4, 7, 3, generator=generator)
    k = torch.randn(1, 2, 7, 3, generator=generator).repeat_interleave(2, dim=1)
    v = torch.randn(1, 2, 7, 3, generator=generator).repeat_interleave(2, dim=1)
    clean = torch.randn(1, 2, 7, 3, generator=generator).repeat_interleave(2, dim=1)
    allowed = torch.ones(7, 7, dtype=torch.bool).tril()
    mask = None if mask_kind == 'none' else allowed[None, None]
    if mask_kind == 'additive':
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float('-inf'))
    sources, queries = [2, 3, 4], [4, 6]
    scale = 3 ** -.5
    current = selected_attention(torch, q, k, v, list(range(7)), mask, scale)
    restored = edge_output(torch, current, q, k, v, clean, sources, queries, mask, scale)
    changed = v.clone()
    changed[:, :, sources, :] = clean[:, :, sources, :]
    expected = selected_attention(torch, q, k, changed, list(range(7)), mask, scale)
    torch.testing.assert_close(restored[:, queries], expected[:, queries], atol=1e-6, rtol=1e-5)
    others = [0, 1, 2, 3, 5]
    assert torch.equal(restored[:, others], current[:, others])
    # Exhaustive query restoration equals replacing source V in ordinary attention.
    all_restored = edge_output(torch, current, q, k, v, clean, sources, list(range(7)), mask, scale)
    torch.testing.assert_close(all_restored, expected, atol=1e-6, rtol=1e-5)
    # Clean self restore is exact; causally earlier queries cannot see audio.
    assert torch.equal(edge_output(torch, current, q, k, v, v, sources, queries, mask, scale), current)
    assert torch.equal(all_restored[:, :2], current[:, :2])
    # Local fixed-attention query partitions are additive (end-to-end effects need not be).
    left = edge_output(torch, current, q, k, v, clean, sources, list(range(6)), mask, scale)
    right = edge_output(torch, current, q, k, v, clean, sources, [6], mask, scale)
    torch.testing.assert_close(left + right - current, all_restored, atol=1e-6, rtol=1e-5)


def test_current_qk_weights_are_used():
    torch = pytest.importorskip('torch')
    # Deliberately nonuniform current routing; clean routing is never provided.
    q = torch.tensor([[[[0.], [4.], [4.]]]])
    k = torch.tensor([[[[1.], [-1.], [0.]]]])
    v = torch.zeros_like(k)
    clean = torch.tensor([[[[2.], [10.], [0.]]]])
    current = torch.zeros(1, 3, 1)
    actual = edge_output(torch, current, q, k, v, clean, [0, 1], [2], None, 1.)
    weights = torch.softmax(torch.tensor([4., -4., 0.]), dim=0)
    torch.testing.assert_close(actual[0, 2, 0], weights[0]*2 + weights[1]*10)
    assert torch.equal(actual[:, :2], current[:, :2])


def test_direct_script_entrypoint_without_repository_on_pythonpath(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'scripts/run_audio_edge_restore.py'
    result=subprocess.run([sys.executable,str(script),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--manifest' in result.stdout
