#!/usr/bin/env python3
"""Scale natural audio-V to decision attention contributions, without donor patches."""
from __future__ import annotations
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
if __package__:
    from . import run_audio_edge_restore as edge
else:
    import run_audio_edge_restore as edge
base = edge.base
LOGGER = logging.getLogger(__name__)
ALPHAS = (0., .5, 1., 1.5, 2.)


def gain_output(torch, current, q, k, v, sources, decision, mask, scaling, alpha):
    """Keep current routing and add (alpha-1) times audio contribution at decision."""
    if not math.isfinite(alpha):
        raise ValueError('Finite gain required')
    if alpha == 1.:
        return current.clone()
    audio_v = torch.zeros_like(v)
    audio_v[:, :, sources, :] = v[:, :, sources, :]
    contribution = edge.selected_attention(torch, q, k, audio_v, [decision], mask, scaling)
    result = current.clone()
    result[:, decision:decision+1, :] += (alpha-1.) * contribution
    return result


def run(args):
    torch, model, tokenizer, whisper = base.load_model(args)
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    model.requires_grad_(False)
    if model.llm.config._attn_implementation != 'sdpa':
        raise ValueError('Expected unchanged SDPA attention implementation')
    rows = base.read_rows(args.manifest, args.limit)[args.start:]
    pairs = base._pair_indices(rows)
    if not pairs:
        raise ValueError('No complete matched pairs')
    layers = base._llm_layers(model)
    if len(layers) != 24:
        raise ValueError('Expected 24 decoder blocks')
    condition = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    ids = [tokenizer.encode(condition[k], add_special_tokens=False) for k in ('happy_verbalizer', 'sad_verbalizer')]
    if ids != [[6247], [12421]]:
        raise ValueError(f'Unexpected single-token verbalizers {ids}')
    prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    sources = base.source_partitions(positions)['audio']
    decision = positions['decision_position']
    device = next(model.parameters()).device
    checks = dict(clean_candidate_parity_max=0., alpha1_noop_max=0., attention_reconstruction_max=0., nondecision_rows_max=0.)
    metadata_keys = ['sample_id', 'pair_id', 'actor', 'statement', 'repetition', 'intensity', 'emotion']
    fields = metadata_keys + ['alpha', 'clean_margin', 'margin', 'shift_from_clean', 'edge_contribution']
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for index, row in enumerate(rows):
            prefix = prefixes[index].unsqueeze(0).to(device)
            mask = torch.ones(prefix.shape[:2], device=device, dtype=torch.bool)

            def forward(alpha=None):
                handles, cache = [], {}
                def capture_inputs(layer):
                    def hook(_module, _args, kwargs):
                        if kwargs.get('past_key_values') is not None:
                            raise ValueError('Requires cache-free forward')
                        cache[layer] = dict(position_embeddings=kwargs['position_embeddings'], attention_mask=kwargs.get('attention_mask'))
                    return hook
                def capture_projection(layer, name):
                    def hook(_module, _inputs, output):
                        cache[layer][name] = output
                    return hook
                def scale_edge(layer):
                    def hook(_module, inputs):
                        state, attn = cache[layer], layers[layer].self_attn
                        if attn.training or attn.sliding_window is not None:
                            raise ValueError('Requires eval attention without sliding window')
                        def reshape(value):
                            return value.view(value.shape[0], value.shape[1], -1, attn.head_dim).transpose(1, 2)
                        q, k = apply_rotary_pos_emb(reshape(state['q']), reshape(state['k']), *state['position_embeddings'])
                        k = repeat_kv(k, attn.num_key_value_groups)
                        v = repeat_kv(reshape(state['v']), attn.num_key_value_groups)
                        original = inputs[0]
                        reconstructed = edge.selected_attention(torch, q, k, v, [decision], state['attention_mask'], attn.scaling)
                        error = float((reconstructed-original[:, decision:decision+1, :]).abs().max())
                        if not math.isfinite(error):
                            raise RuntimeError('Non-finite attention reconstruction')
                        checks['attention_reconstruction_max'] = max(checks['attention_reconstruction_max'], error)
                        changed = gain_output(torch, original, q, k, v, sources, decision, state['attention_mask'], attn.scaling, alpha)
                        others = [i for i in range(original.shape[1]) if i != decision]
                        if not bool(torch.isfinite(changed).all()):
                            raise RuntimeError('Non-finite gain output')
                        outside = float((changed[:, others]-original[:, others]).abs().max())
                        checks['nondecision_rows_max'] = max(checks['nondecision_rows_max'], outside)
                        return (changed,) + inputs[1:]
                    return hook
                try:
                    if alpha is not None:
                        for layer in base.RESTORE_LAYERS:
                            attn = layers[layer].self_attn
                            handles.append(attn.register_forward_pre_hook(capture_inputs(layer), with_kwargs=True))
                            for name in ('q', 'k', 'v'):
                                handles.append(getattr(attn, name+'_proj').register_forward_hook(capture_projection(layer, name)))
                            handles.append(attn.o_proj.register_forward_pre_hook(scale_edge(layer)))
                    with torch.inference_mode():
                        result = base._decoder_forward(model, prefix, mask)
                        logits = model.llm.lm_head(result.last_hidden_state[:, decision, :]).float()
                        value = float(logits[0, 6247]-logits[0, 12421])
                        if not math.isfinite(value):
                            raise RuntimeError('Non-finite margin')
                        return value
                finally:
                    for handle in handles:
                        handle.remove()

            clean = forward()
            happy, sad = base._score_candidate_pairs(torch, model, prefix, mask, ids[0], ids[1], positions['prefix_length'])
            error = abs(float(happy[0]-sad[0])-clean)
            if not math.isfinite(error):
                raise RuntimeError('Non-finite candidate parity')
            checks['clean_candidate_parity_max'] = max(checks['clean_candidate_parity_max'], error)
            for alpha in ALPHAS:
                margin = forward(alpha)
                if alpha == 1.:
                    checks['alpha1_noop_max'] = max(checks['alpha1_noop_max'], abs(margin-clean))
                writer.writerow(dict({key: row[key] for key in metadata_keys}, alpha=alpha, clean_margin=clean,
                                     margin=margin, shift_from_clean=margin-clean, edge_contribution=clean-margin))
            stream.flush()
            if checks['alpha1_noop_max'] != 0. or checks['nondecision_rows_max'] != 0.:
                raise RuntimeError(f'Exact no-op/isolation check failed: {checks}')
            if any(not math.isfinite(v) or v > args.self_patch_tolerance for v in checks.values()):
                raise RuntimeError(f'Numerical validation failed: {checks}')
            LOGGER.info('completed sample %d/%d checks=%s', index+1, len(rows), checks)
    meta = dict(n_samples=len(rows), n_pairs=len(pairs), seed=args.seed, positions=positions,
                source_positions=sources, query_positions=[decision], source_spans_half_open=base._spans(sources),
                condition_id='upper_prompt__lower_spaced', prompt=condition['prompt'], happy_token_ids=ids[0], sad_token_ids=ids[1],
                checkpoint=str(args.checkpoint), model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                torch_version=torch.__version__, transformers_version=transformers.__version__,
                attention_implementation=model.llm.config._attn_implementation, model_dtype=str(next(model.parameters()).dtype),
                numerical_checks=checks, tolerance=args.self_patch_tolerance, candidate_parity_samples=len(rows), noop_samples=len(rows),
                upstream_patch=None, donor_used=False, gain_layers=base.RESTORE_LAYERS, alphas=ALPHAS,
                intervention='pre-o_proj decision row += (alpha-1) * current attention weights @ audio-only current V; no attention renormalization; downstream states evolve naturally',
                edge_contribution_definition='clean_margin - margin; natural edge necessity C(x) uses alpha=0 only',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(edge.__file__), Path(base.__file__))})
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(base.parse_args())
