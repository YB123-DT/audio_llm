"""Block schedules and reuse of the verified source-query edge intervention."""
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import run_audio_edge_restore as edge
from scripts import run_block_direct_edge_restore as blocks


def test_unique_single_and_cumulative_schedules():
    schedules = blocks.restore_schedules()
    assert len(schedules) == 11
    assert len({tuple(v) for v in schedules.values()}) == 11
    for layer in range(18, 24):
        assert schedules[f'direct_block_{layer}'] == [layer]
    for end in range(19, 24):
        assert schedules[f'direct_cumulative_18_{end}'] == list(range(18, end + 1))
    assert 'direct_cumulative_18' not in schedules
    assert all(17 not in value for value in schedules.values())


def test_reuses_verified_edge_primitive():
    assert blocks.edge_output is edge.edge_output
    assert blocks.selected_attention is edge.selected_attention


def test_decision_restore_keeps_every_other_query_unchanged():
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(321)
    q, k, v, clean = [torch.randn(1, 2, 5, 3, generator=generator) for _ in range(4)]
    original = blocks.selected_attention(torch, q, k, v, list(range(5)), None, 3**-.5)
    actual = blocks.edge_output(torch, original, q, k, v, clean, [1, 2], [4], None, 3**-.5)
    mixed = v.clone()
    mixed[:, :, [1, 2]] = clean[:, :, [1, 2]]
    expected = blocks.selected_attention(torch, q, k, mixed, [4], None, 3**-.5)
    torch.testing.assert_close(actual[:, [4]], expected)
    assert torch.equal(actual[:, :4], original[:, :4])


def test_direct_entrypoint_without_repository_on_pythonpath(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_block_direct_edge_restore.py'
    result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--manifest' in result.stdout
