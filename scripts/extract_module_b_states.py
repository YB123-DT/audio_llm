#!/usr/bin/env python3
"""Extract clean, complete decoder states with Module A endpoint parity gates."""
from __future__ import annotations
import argparse
import csv
import hashlib
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
    if len(ids) != len(set(ids)) or set(ids) != set(sample_ids):
        raise ValueError('Reference sample IDs must uniquely match extraction IDs')
    order = [ids.index(s) for s in sample_ids]
    return {k: reference[k][order] for k in ('projector', 'answer_state', 'clean_margins')}


def pooled_states(hidden, audio_start, audio_length, decision):
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError('Expected one complete sequence per forward')
    if not 0 <= audio_start < audio_start + audio_length <= decision < hidden.shape[1]:
        raise ValueError('Invalid audio/answer positions')
    return hidden[0, audio_start:audio_start + audio_length].mean(dim=0), hidden[0, decision]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'ravdess-root', 'slam-llm-root', 'qwen-path', 'whisper-path', 'checkpoint', 'output', 'module-a-vectors', 'metadata'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.seed = 1234
    torch, model, tokenizer, whisper = base.load_model(args)
    model.requires_grad_(False)
    rows = base.read_rows(args.manifest, None)
    sample_ids = [r['sample_id'] for r in rows]
    if len(rows) != 192 or len(set(sample_ids)) != 192:
        raise ValueError('Module B requires the complete 192 unique samples')
    with np.load(args.module_a_vectors) as archive:
        reference = aligned_reference(archive, sample_ids)
    layers = base._llm_layers(model)
    if len(layers) != 24:
        raise ValueError('Expected 24 decoder blocks')
    condition = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    if [tokenizer.encode(condition[k], add_special_tokens=False) for k in ('happy_verbalizer', 'sad_verbalizer')] != [[6247], [12421]]:
        raise ValueError('Single-token verbalizers changed')
    projected = []
    def capture_projector(_module, _inputs, output):
        projected.append(output[0].mean(dim=0).detach().float().cpu().numpy())
    handle = model.encoder_projector.register_forward_hook(capture_projector)
    try:
        prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    finally:
        handle.remove()
    if len(projected) != 192:
        raise ValueError('Projector capture count mismatch')
    device = next(model.parameters()).device
    audio, answer, final, margins = [], [], [], []
    with torch.inference_mode():
        d_lm = (model.llm.lm_head.weight[6247] - model.llm.lm_head.weight[12421]).float().cpu().numpy()
        for i, prefix_cpu in enumerate(prefixes):
            captured, handles = {}, []
            def capture(layer):
                def hook(_module, _inputs, output):
                    h = base._hidden_tensor(output)
                    a, d = pooled_states(h, positions['audio_start'], positions['audio_length'], positions['decision_position'])
                    captured[layer] = (a.detach().float().cpu().numpy(), d.detach().float().cpu().numpy())
                return hook
            try:
                for l, block in enumerate(layers):
                    handles.append(block.register_forward_hook(capture(l)))
                prefix = prefix_cpu.unsqueeze(0).to(device)
                result = base._decoder_forward(model, prefix, torch.ones(prefix.shape[:2], dtype=torch.bool, device=device))
                h = result.last_hidden_state[0, positions['decision_position']]
                logits = model.llm.lm_head(h).float()
                margins.append(float(logits[6247] - logits[12421]))
                final.append(h.detach().float().cpu().numpy())
            finally:
                for hndl in handles:
                    hndl.remove()
            if len(captured) != 24:
                raise ValueError('Missing decoder blocks')
            audio.append(np.stack([captured[l][0] for l in range(24)]))
            answer.append(np.stack([captured[l][1] for l in range(24)]))
            logging.info('Completed clean decoder %d/192', i + 1)
    arrays = dict(sample_ids=np.array(sample_ids), projector=np.stack(projected), audio_states=np.stack(audio), answer_states=np.stack(answer), answer_final_norm=np.stack(final), clean_margins=np.array(margins), d_lm=d_lm)
    checks = {k: float(np.max(np.abs(arrays[a] - reference[b]))) for k, a, b in [('module_a_projector_max', 'projector', 'projector'), ('module_a_answer_max', 'answer_final_norm', 'answer_state'), ('module_a_margin_max', 'clean_margins', 'clean_margins')]}
    checks['norm_head_margin_max'] = float(np.max(np.abs(arrays['answer_final_norm'] @ d_lm - arrays['clean_margins'])))
    norm = model.llm.model.norm
    with torch.inference_mode():
        rebuilt = norm(torch.from_numpy(arrays['answer_states'][:, -1]).to(device)).cpu().numpy()
    checks['postblock23_finalnorm_max'] = float(np.max(np.abs(rebuilt - arrays['answer_final_norm'])))
    for name, value in checks.items():
        if not np.isfinite(value) or value > 1e-3:
            raise ValueError(f'Endpoint parity gate failed: {name}={value}')
    for name, value in arrays.items():
        if name != 'sample_ids' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite {name}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    (args.output.parent / 'metadata.csv').write_bytes(args.metadata.read_bytes())
    import transformers
    files = [Path(__file__), Path(base.__file__), args.module_a_vectors, args.metadata, args.manifest, args.qwen_path / 'config.json']
    provenance = dict(seed=1234, n_samples=192, layers=list(range(24)), shapes={k:list(v.shape) for k,v in arrays.items()}, positions=positions, checks=checks, tolerance=1e-3,
        prompt=condition['prompt'], happy_token_id=6247, sad_token_id=12421, model_parameters_frozen=all(not p.requires_grad for p in model.parameters()), decoder_forwards_per_sample=1,
        state_semantics='audio_states and answer_states are complete post-block residuals (0-based blocks 0..23), before terminal RMSNorm. answer_final_norm is separately normalized final answer position. Projector is mean before decoder.',
        cache_policy='Fresh extraction required: Module A rejected historical complete-answer cache with margin max error 0.7586999193; cause not established. Existing compatible endpoints reused as parity references, no intervention.',
        checkpoint=str(args.checkpoint), torch_version=torch.__version__, transformers_version=transformers.__version__, dtype=str(next(model.parameters()).dtype),
        sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
