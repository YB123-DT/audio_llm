#!/usr/bin/env python3
"""Extract per-block natural audio-source updates at the decision query, without intervention."""
from __future__ import annotations
import csv
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
if __package__:
    from . import run_audio_edge_restore as edge
else:
    import run_audio_edge_restore as edge
base = edge.base
LOGGER = logging.getLogger(__name__)


def source_update(torch, q, k, v, sources, decision, mask, scaling, weight):
    """Original attention weights, selected values, o_proj without shared bias."""
    selected = torch.zeros_like(v)
    selected[:, :, sources, :] = v[:, :, sources, :]
    pre = edge.selected_attention(torch, q, k, selected, [decision], mask, scaling)
    return torch.nn.functional.linear(pre, weight, bias=None)


def run(args):
    torch, model, tokenizer, whisper = base.load_model(args)
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    model.requires_grad_(False)
    if model.llm.config._attn_implementation != 'sdpa':
        raise ValueError('Expected SDPA')
    rows = base.read_rows(args.manifest, args.limit)[args.start:]
    pairs = base._pair_indices(rows)
    if not pairs:
        raise ValueError('Complete matched pairs required')
    layers = base._llm_layers(model)
    if len(layers) != 24:
        raise ValueError('Expected 24 layers')
    condition = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    ids = [tokenizer.encode(condition[k], add_special_tokens=False) for k in ('happy_verbalizer', 'sad_verbalizer')]
    if ids != [[6247], [12421]]:
        raise ValueError(f'Unexpected verbalizers {ids}')
    prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    groups = base.source_partitions(positions)
    decision = positions['decision_position']
    device = next(model.parameters()).device
    checks = dict(attention_reconstruction_max=0., source_decomposition_max=0., hooked_clean_parity_max=0., candidate_parity_max=0., historical_clean_parity_max=0., norm_readout_parity_max=0.)
    reference_path = Path('reports/2026-09-15-clean-edge/gain.csv')
    with reference_path.open() as f:
        reference = {r['sample_id']: float(r['clean_margin']) for r in csv.DictReader(f) if float(r['alpha']) == 1.}
    vectors, margins, pre_norms = [], [], []
    norm = model.llm.model.norm
    d_lm = (model.llm.lm_head.weight[6247]-model.llm.lm_head.weight[12421]).detach()
    def check(name, value):
        if not np.isfinite(value) or value > args.self_patch_tolerance:
            raise RuntimeError(f'{name}: {value}')
        checks[name] = max(checks[name], value)
    for index, row in enumerate(rows):
        prefix = prefixes[index].unsqueeze(0).to(device)
        mask = torch.ones(prefix.shape[:2], device=device, dtype=torch.bool)
        handles, cache, updates, final = [], {}, {}, {}
        def capture_inputs(layer):
            def hook(_module, _args, kwargs):
                if kwargs.get('past_key_values') is not None:
                    raise ValueError('Cache-free extraction required')
                cache[layer] = dict(position_embeddings=kwargs['position_embeddings'], attention_mask=kwargs.get('attention_mask'))
            return hook
        def capture_projection(layer, name):
            def hook(_module, _inputs, output):
                cache[layer][name] = output
            return hook
        def capture_edge(layer):
            def hook(module, inputs):
                attn, state = layers[layer].self_attn, cache[layer]
                if attn.training or attn.sliding_window is not None:
                    raise ValueError('Eval attention without sliding window required')
                def reshape(v):
                    return v.view(v.shape[0], v.shape[1], -1, attn.head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(reshape(state['q']), reshape(state['k']), *state['position_embeddings'])
                k = repeat_kv(k, attn.num_key_value_groups)
                v = repeat_kv(reshape(state['v']), attn.num_key_value_groups)
                rebuilt = edge.selected_attention(torch, q, k, v, [decision], state['attention_mask'], attn.scaling)
                original = inputs[0][:, decision:decision+1]
                check('attention_reconstruction_max', float((rebuilt-original).abs().max()))
                audio = source_update(torch, q, k, v, groups['audio'], decision, state['attention_mask'], attn.scaling, module.weight)
                other = source_update(torch, q, k, v, groups['non_audio'], decision, state['attention_mask'], attn.scaling, module.weight)
                full = torch.nn.functional.linear(original, module.weight, bias=None)
                check('source_decomposition_max', float((audio+other-full).abs().max()))
                if not bool(torch.isfinite(audio).all()):
                    raise RuntimeError('Nonfinite edge vectors')
                updates[layer] = audio[0, 0].float().cpu().numpy()
                # Observational hook: return None so the original computation is untouched.
            return hook
        def capture_norm(_module, inputs):
            final['pre_norm'] = inputs[0][:, decision].detach().clone()
        with torch.inference_mode():
            clean = base._decoder_forward(model, prefix, mask)
            logits = model.llm.lm_head(clean.last_hidden_state[:, decision]).float()
            clean_margin = float(logits[0, 6247]-logits[0, 12421])
            try:
                for layer in base.RESTORE_LAYERS:
                    attn = layers[layer].self_attn
                    handles.append(attn.register_forward_pre_hook(capture_inputs(layer), with_kwargs=True))
                    for name in ('q', 'k', 'v'):
                        handles.append(getattr(attn, name+'_proj').register_forward_hook(capture_projection(layer, name)))
                    handles.append(attn.o_proj.register_forward_pre_hook(capture_edge(layer)))
                handles.append(norm.register_forward_pre_hook(capture_norm))
                hooked = base._decoder_forward(model, prefix, mask)
                logits = model.llm.lm_head(hooked.last_hidden_state[:, decision]).float()
                hooked_margin = float(logits[0, 6247]-logits[0, 12421])
            finally:
                for handle in handles:
                    handle.remove()
            check('hooked_clean_parity_max', abs(clean_margin-hooked_margin))
            check('historical_clean_parity_max', abs(clean_margin-reference[row['sample_id']]))
            happy, sad = base._score_candidate_pairs(torch, model, prefix, mask, ids[0], ids[1], positions['prefix_length'])
            check('candidate_parity_max', abs(float(happy[0]-sad[0])-clean_margin))
            check('norm_readout_parity_max', abs(float((norm(final['pre_norm'])*d_lm).sum())-clean_margin))
        vectors.append(np.stack([updates[l] for l in base.RESTORE_LAYERS]))
        margins.append(clean_margin)
        pre_norms.append(final['pre_norm'][0].float().cpu().numpy())
        LOGGER.info('completed sample %d/%d checks=%s', index+1, len(rows), checks)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['sample_id', 'pair_id', 'actor', 'statement', 'repetition', 'intensity', 'emotion', 'clean_margin']
    with args.output_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(dict({k:r[k] for k in fields[:-1]}, clean_margin=m) for r,m in zip(rows,margins))
    np.savez_compressed(args.output_csv.with_suffix('.npz'), edge_updates=np.stack(vectors), layers=base.RESTORE_LAYERS,
                        sample_ids=np.array([r['sample_id'] for r in rows]), clean_margins=margins,
                        d_lm=d_lm.float().cpu().numpy(), finalnorm_weight=norm.weight.detach().float().cpu().numpy(),
                        d_lm_norm_weighted=(d_lm*norm.weight).detach().float().cpu().numpy(), final_decision_pre_norm=np.stack(pre_norms))
    meta = dict(n_samples=len(rows), n_pairs=len(pairs), layers=base.RESTORE_LAYERS, positions=positions,
                numerical_checks=checks, tolerance=args.self_patch_tolerance, donor_used=False, intervention=None,
                vector_definition='Per-layer W_o @ sum_audio A(decision,a) V(a); original clean QK and softmax denominator, o_proj bias excluded, no cross-layer sum',
                lm_direction_note='Raw LM-head difference is not an effective earlier-layer gradient; norm-weighted direction only accounts for final RMSNorm gain, not intervening blocks or input-dependent normalization',
                prompt=condition['prompt'], happy_token_id=6247, sad_token_id=12421, checkpoint=str(args.checkpoint),
                model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                torch_version=torch.__version__, transformers_version=transformers.__version__, model_dtype=str(next(model.parameters()).dtype),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(edge.__file__),Path(base.__file__))},
                historical_reference=str(reference_path), historical_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(base.parse_args())
