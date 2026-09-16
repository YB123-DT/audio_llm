import pytest
torch = pytest.importorskip('torch')
from scripts.run_early_answer_ablation import suppress_answer_update, install_hooks


def test_answer_update_tensor_and_tuple_isolation():
    original = torch.randn(1, 4, 8)
    snapshot = original.clone()
    for output in (original, (original, 'aux')):
        changed = suppress_answer_update(output, 3)
        hidden = changed[0] if isinstance(changed, tuple) else changed
        assert torch.equal(original, snapshot)
        assert torch.equal(hidden[:, :3], original[:, :3])
        assert torch.count_nonzero(hidden[:, 3]) == 0
        if isinstance(changed, tuple):
            assert changed[1] == 'aux'
    with pytest.raises(ValueError):
        suppress_answer_update(original, 2)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Linear(8, 8)
        self.mlp = torch.nn.Linear(8, 8)
    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


def test_blocks_components_and_nonanswer_isolation():
    torch.manual_seed(1234)
    blocks = torch.nn.ModuleList([Block() for _ in range(8)])
    x = torch.randn(1, 4, 8)
    clean, checks = {}, {'nonanswer_block_comparisons':0}
    outputs = {}
    for condition in ('clean','no_attn','no_mlp'):
        captured = {}
        handles, calls = install_hooks(blocks,condition,3,captured,clean,checks)
        out = x.clone()
        try:
            for block in blocks:
                out = block(out)
        finally:
            for handle in handles:
                handle.remove()
        manual = x.clone()
        for layer, block in enumerate(blocks):
            update = block.self_attn(manual)
            if condition == 'no_attn' and 1 <= layer <= 6:
                update = suppress_answer_update(update, 3)
            manual = manual + update
            update = block.mlp(manual)
            if condition == 'no_mlp' and 1 <= layer <= 6:
                update = suppress_answer_update(update, 3)
            manual = manual + update
        assert torch.equal(out, manual)
        assert calls == ([] if condition == 'clean' else list(range(1,7)))
        assert captured['block6'].shape == (8,)
        outputs[condition] = out
    assert checks['nonanswer_block_comparisons'] == 16
    assert not torch.equal(outputs['clean'][:,3],outputs['no_attn'][:,3])
    assert not torch.equal(outputs['clean'][:,3],outputs['no_mlp'][:,3])
    assert all(not b._forward_hooks and not b.self_attn._forward_hooks and not b.mlp._forward_hooks for b in blocks)
