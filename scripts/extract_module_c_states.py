#!/usr/bin/env python3
"""Observe complete answer residuals before/after each natural decoder sublayer."""
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
except ImportError:
    import run_activation_patching as base


def aligned_reference(reference, sample_ids):
    ids = reference['sample_ids'].astype(str).tolist()
    if len(sample_ids) != len(set(sample_ids)) or len(ids) != len(set(ids)) or set(ids) != set(sample_ids):
        raise ValueError('Reference sample IDs must uniquely match extraction IDs')
    order = [ids.index(s) for s in sample_ids]
    return {k: reference[k][order] for k in ('answer_states', 'answer_final_norm', 'clean_margins')}


def install_observers(blocks, decision, captured):
    """Return removable read-only hooks; CPU copies prevent later alias changes."""
    handles = []
    def record(layer, name, hidden):
        h = base._hidden_tensor(hidden)
        if h.ndim != 3 or h.shape[0] != 1 or not 0 <= decision < h.shape[1]:
            raise ValueError('Expected one sequence and a valid answer position')
        captured.setdefault(layer, {})[name] = h[0, decision].detach().float().cpu().numpy().copy()
    def pre(layer, name):
        def hook(_module, inputs):
            record(layer, name, inputs[0])
        return hook
    def post(layer, name):
        def hook(_module, _inputs, output):
            record(layer, name, output)
        return hook
    for layer, block in enumerate(blocks):
        handles.extend([
            block.input_layernorm.register_forward_pre_hook(pre(layer, 'in')),
            block.post_attention_layernorm.register_forward_pre_hook(pre(layer, 'attn')),
            block.register_forward_hook(post(layer, 'out')),
            block.self_attn.register_forward_hook(post(layer, 'attention_update')),
            block.mlp.register_forward_hook(post(layer, 'mlp_update')),
        ])
    return handles


def parity_checks(arrays, reference):
    return {
        'module_b_all_block_outputs_max': float(np.max(np.abs(arrays['answer_out'] - reference['answer_states']))),
        'module_b_final_norm_max': float(np.max(np.abs(arrays['answer_final_norm'] - reference['answer_final_norm']))),
        'module_b_margin_max': float(np.max(np.abs(arrays['clean_margins'] - reference['clean_margins']))),
        'adjacent_block_residual_max': float(np.max(np.abs(arrays['answer_in'][:, 1:] - arrays['answer_out'][:, :-1]))),
        'block0_input_between_samples_max': float(np.max(np.abs(arrays['answer_in'][:, 0] - arrays['answer_in'][0, 0]))),
    }


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
    sample_ids = [r['sample_id'] for r in rows]
    if len(rows) != 192 or len(set(sample_ids)) != 192:
        raise ValueError('Module C requires 192 unique samples')
    with np.load(args.module_b_vectors) as archive:
        reference = aligned_reference(archive, sample_ids)
    blocks = base._llm_layers(model)
    if len(blocks) != 24:
        raise ValueError('Expected 24 decoder blocks')
    source = inspect.getsource(type(blocks[0]).forward)
    source_path = Path(inspect.getfile(type(blocks[0])))
    condition = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    if [tokenizer.encode(condition[k], add_special_tokens=False) for k in ('happy_verbalizer', 'sad_verbalizer')] != [[6247], [12421]]:
        raise ValueError('Single-token verbalizers changed')
    prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    device = next(model.parameters()).device
    states = {k: [] for k in ('in', 'attn', 'out')}
    finals, margins = [], []
    addition_errors = {'attention_residual_add_max': 0., 'mlp_residual_add_max': 0.}
    with torch.inference_mode():
        d_lm = (model.llm.lm_head.weight[6247] - model.llm.lm_head.weight[12421]).float().cpu().numpy()
        for i, prefix_cpu in enumerate(prefixes):
            captured = {}
            handles = install_observers(blocks, positions['decision_position'], captured)
            try:
                prefix = prefix_cpu.unsqueeze(0).to(device)
                result = base._decoder_forward(model, prefix, torch.ones(prefix.shape[:2], dtype=torch.bool, device=device))
                h = result.last_hidden_state[0, positions['decision_position']]
                logits = model.llm.lm_head(h).float()
                margins.append(float(logits[6247] - logits[12421]))
                finals.append(h.detach().float().cpu().numpy())
            finally:
                for handle in handles:
                    handle.remove()
            if len(captured) != 24 or any(len(v) != 5 for v in captured.values()):
                raise ValueError('Incomplete sublayer capture')
            for k in states:
                states[k].append(np.stack([captured[l][k] for l in range(24)]))
            for c in captured.values():
                for name, start, update, end in [('attention_residual_add_max', 'in', 'attention_update', 'attn'), ('mlp_residual_add_max', 'attn', 'mlp_update', 'out')]:
                    addition_errors[name] = max(addition_errors[name], float(np.max(np.abs(c[start] + c[update] - c[end]))))
            logging.info('Completed normal decoder %d/192', i + 1)
    arrays = dict(sample_ids=np.array(sample_ids), **{'answer_' + k: np.stack(v) for k, v in states.items()}, answer_final_norm=np.stack(finals), clean_margins=np.array(margins), d_lm=d_lm)
    checks = parity_checks(arrays, reference)
    checks.update(addition_errors)
    # Identical normal forward must reproduce every cached block, not just endpoints.
    if any(not np.isfinite(v) or v != 0 for v in checks.values()):
        raise ValueError(f'Exact observational parity gate failed: {checks}')
    checks['norm_head_margin_max'] = float(np.max(np.abs(arrays['answer_final_norm'] @ d_lm - arrays['clean_margins'])))
    if checks['norm_head_margin_max'] > 1e-3:
        raise ValueError('LM head reconstruction failed')
    for k, v in arrays.items():
        if k != 'sample_ids' and not np.isfinite(v).all():
            raise ValueError(f'Nonfinite {k}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    import transformers
    files = [Path(__file__), Path(base.__file__), args.module_b_vectors, args.metadata, args.manifest, args.qwen_path / 'config.json', source_path]
    provenance = dict(seed=1234, n_samples=192, layers=list(range(24)), shapes={k:list(v.shape) for k,v in arrays.items()}, positions=positions, checks=checks,
        prompt=condition['prompt'], happy_token_id=6247, sad_token_id=12421, model_parameters_frozen=all(not p.requires_grad for p in model.parameters()), decoder_forwards_per_sample=1,
        state_semantics={'answer_in':'Complete residual before input_layernorm', 'answer_attn':'Complete residual after attention add, before post_attention_layernorm', 'answer_out':'Complete residual after MLP add, before next block/terminal RMSNorm', 'answer_final_norm':'Complete terminal RMSNorm answer state'},
        intervention='None; hooks only read/copy and return None', checkpoint=str(args.checkpoint), torch_version=torch.__version__, transformers_version=transformers.__version__, dtype=str(next(model.parameters()).dtype), decoder_forward_source=source,
        sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
