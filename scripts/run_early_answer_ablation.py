#!/usr/bin/env python3
"""Three fixed early answer-only update ablations; no model weights change."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import logging
from pathlib import Path
import numpy as np
try:
    from . import run_activation_patching as base
    from .extract_module_c_states import aligned_reference
except ImportError:
    import run_activation_patching as base
    from extract_module_c_states import aligned_reference

CONDITIONS = ('clean', 'no_attn', 'no_mlp')
TARGET_BLOCKS = tuple(range(1, 7))


def suppress_answer_update(output, decision):
    """Clone output and zero just one residual-update row, preserving tuple extras."""
    hidden = base._hidden_tensor(output)
    if hidden.ndim != 3 or hidden.shape[0] != 1 or decision != hidden.shape[1] - 1:
        raise ValueError('Expected batch one with answer at final position')
    altered = hidden.clone()
    altered[:, decision, :] = 0
    if not altered[:, :decision].equal(hidden[:, :decision]) or altered[:, decision].count_nonzero().item():
        raise AssertionError('Update row isolation failed')
    return (altered, *output[1:]) if isinstance(output, tuple) else altered


def install_hooks(blocks, condition, decision, captured, clean_nonanswer, checks):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    handles = []
    calls = []
    def suppress(layer):
        def hook(_module, _inputs, output):
            calls.append(layer)
            return suppress_answer_update(output, decision)
        return hook
    def observe(layer):
        def hook(_module, _inputs, output):
            hidden = base._hidden_tensor(output)
            if hidden.shape[1] != decision + 1:
                raise ValueError('Answer must be last position for causal isolation')
            nonanswer = hidden[:, :decision].detach()
            if condition == 'clean':
                clean_nonanswer[layer] = nonanswer.clone()
            elif not nonanswer.equal(clean_nonanswer[layer]):
                raise AssertionError(f'Non-answer state changed at block {layer}')
            checks['nonanswer_block_comparisons'] += condition != 'clean'
            if layer == 6:
                captured['block6'] = hidden[0, decision].detach().float().cpu().numpy().copy()
        return hook
    for layer, block in enumerate(blocks):
        if condition != 'clean' and layer in TARGET_BLOCKS:
            component = block.self_attn if condition == 'no_attn' else block.mlp
            handles.append(component.register_forward_hook(suppress(layer)))
        handles.append(block.register_forward_hook(observe(layer)))
    return handles, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'ravdess-root', 'slam-llm-root', 'qwen-path', 'whisper-path', 'checkpoint', 'output', 'module-b-vectors', 'metadata'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.seed = 1234
    torch, model, tokenizer, whisper = base.load_model(args)
    model.requires_grad_(False)
    rows = base.read_rows(args.manifest, None)
    ids = [r['sample_id'] for r in rows]
    if len(ids) != 192 or len(set(ids)) != 192:
        raise ValueError('Require exactly 192 unique samples')
    with np.load(args.module_b_vectors) as archive:
        reference = aligned_reference(archive, ids)
    blocks = base._llm_layers(model)
    if len(blocks) != 24:
        raise ValueError('Expected 24 blocks')
    verbalizer = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    if [tokenizer.encode(verbalizer[k], add_special_tokens=False) for k in ('happy_verbalizer','sad_verbalizer')] != [[6247],[12421]]:
        raise ValueError('Verbalizer token mismatch')
    prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, verbalizer['prompt'])
    decision = positions['decision_position']
    if decision != 330:
        raise ValueError('Unexpected decision position')
    device = next(model.parameters()).device
    block6, finals, margins = [], [], []
    checks = {'nonanswer_block_comparisons': 0, 'suppressed_update_calls': 0}
    with torch.inference_mode():
        d_lm = (model.llm.lm_head.weight[6247] - model.llm.lm_head.weight[12421]).float().cpu().numpy()
        for i, prefix_cpu in enumerate(prefixes):
            prefix = prefix_cpu.unsqueeze(0).to(device)
            mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=device)
            clean_nonanswer, sample_b6, sample_final = {}, [], []
            for condition in CONDITIONS:
                captured = {}
                handles, calls = install_hooks(blocks, condition, decision, captured, clean_nonanswer, checks)
                try:
                    result = base._decoder_forward(model, prefix, mask)
                    final = result.last_hidden_state[0, decision]
                    sample_final.append(final.detach().float().cpu().numpy().copy())
                    sample_b6.append(captured['block6'])
                    if condition == 'clean':
                        logits = model.llm.lm_head(final).float()
                        margins.append(float(logits[6247] - logits[12421]))
                finally:
                    for handle in handles:
                        handle.remove()
                if calls != ([] if condition == 'clean' else list(TARGET_BLOCKS)):
                    raise AssertionError(f'Wrong intervention calls: {calls}')
                checks['suppressed_update_calls'] += len(calls)
            block6.append(np.stack(sample_b6))
            finals.append(np.stack(sample_final))
            logging.info('Completed sample %d/192 (three forwards)', i+1)
    arrays = dict(sample_ids=np.array(ids), conditions=np.array(CONDITIONS), block6=np.stack(block6), final_answer=np.stack(finals), clean_margins=np.array(margins), d_lm=d_lm)
    checks.update(clean_block6_max=float(np.max(np.abs(arrays['block6'][:,0]-reference['answer_states'][:,6]))), clean_final_max=float(np.max(np.abs(arrays['final_answer'][:,0]-reference['answer_final_norm']))), clean_margin_max=float(np.max(np.abs(arrays['clean_margins']-reference['clean_margins']))))
    if any(checks[k] != 0 for k in ('clean_block6_max','clean_final_max','clean_margin_max')):
        raise AssertionError(f'Clean reference mismatch {checks}')
    if checks['nonanswer_block_comparisons'] != 192*2*24 or checks['suppressed_update_calls'] != 192*2*6:
        raise AssertionError('Wrong validation count')
    for key, value in arrays.items():
        if key not in ('sample_ids','conditions') and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite {key}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    import transformers
    source_path = Path(inspect.getfile(type(blocks[0])))
    files = [Path(__file__), Path(base.__file__), args.module_b_vectors, args.metadata, args.manifest, args.qwen_path/'config.json', source_path]
    provenance = dict(seed=1234,n_samples=192,conditions=CONDITIONS,target_blocks=TARGET_BLOCKS,positions=positions,checks=checks,shapes={k:list(v.shape) for k,v in arrays.items()},prompt=verbalizer['prompt'],happy_token_id=6247,sad_token_id=12421,decoder_forwards_per_sample=3,model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),torch_version=torch.__version__,transformers_version=transformers.__version__,dtype=str(next(model.parameters()).dtype),checkpoint=str(args.checkpoint),state_semantics={'block6':'Complete answer post-block6 residual before next layer norm','final_answer':'Complete final answer after terminal RMSNorm'},intervention='Forward output hook zeros only answer row of self_attn or mlp update, before residual addition, at zero-based blocks1..6. Other component computed normally on current residual. All 330 non-answer rows at every block are bitwise equal to clean, including all audio positions.',decoder_forward_source=inspect.getsource(type(blocks[0]).forward),sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
