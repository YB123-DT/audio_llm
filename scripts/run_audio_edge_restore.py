#!/usr/bin/env python3
"""Restore audio-source values for selected queries, retaining current QK weights."""
from __future__ import annotations
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
if __package__:
    from . import run_source_value_restore as base
else:
    import run_source_value_restore as base

LOGGER = logging.getLogger(__name__)


def query_partitions(positions):
    groups = base.source_partitions(positions)
    return dict(decision=groups['decision'], audio=groups['audio'],
                end_marker=groups['post_audio_marker'],
                non_decision=[i for i in groups['all'] if i != positions['decision_position']],
                all=groups['all'])


def selected_attention(torch, q, k, v, queries, mask, scaling):
    """GQA already expanded; causal coordinates refer to original query positions."""
    query = q[:, :, queries, :]
    if mask is None:
        selected_mask = torch.arange(k.shape[-2], device=q.device)[None, :] <= torch.tensor(queries, device=q.device)[:, None]
    else:
        selected_mask = mask[..., :k.shape[-2]]
        if selected_mask.shape[-2] != 1:
            selected_mask = selected_mask[..., queries, :]
    return torch.nn.functional.scaled_dot_product_attention(
        query.contiguous(), k.contiguous(), v.contiguous(), attn_mask=selected_mask,
        dropout_p=0.0, is_causal=False, scale=scaling).transpose(1, 2).flatten(2)


def edge_output(torch, current, q, k, v, clean_v, sources, queries, mask, scaling):
    """Add current attention weights times source-restricted clean-minus-current V."""
    delta = torch.zeros_like(v)
    delta[:, :, sources, :] = clean_v[:, :, sources, :] - v[:, :, sources, :]
    result = current.clone()
    result[:, queries, :] += selected_attention(torch, q, k, delta, queries, mask, scaling)
    return result


def run(args):
    torch, model, tokenizer, whisper = base.load_model(args)
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    model.requires_grad_(False)
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
        raise ValueError(f'Unexpected verbalizers {ids}')
    prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    sources = base.source_partitions(positions)['audio']
    groups = query_partitions(positions)
    device = next(model.parameters()).device
    donors = base._capture_all_donors(torch, model, prefixes, positions, (17,), device, args.batch_size)[17]['audio_tokens']
    counterpart = {}
    for pair in pairs:
        h, s = pair['happy_index'], pair['sad_index']
        counterpart[h], counterpart[s] = s, h
    if len(counterpart) != len(rows):
        raise ValueError('Incomplete counterpart mapping')
    checks = dict(clean_candidate_parity_max=0., patched_candidate_parity_max=0., self_audio_max=0.,
                  component_noop_max=0., attention_reconstruction_max=0., nonselected_rows_max=0.,
                  all_query_source_reference_max=0., pre_audio_query_delta_max=0.)
    metadata_keys = ['sample_id', 'pair_id', 'actor', 'statement', 'repetition', 'intensity', 'emotion']
    fields = metadata_keys + ['donor_sample_id', 'donor_sign', 'condition', 'component', 'restore_layers',
                             'clean_margin', 'patched_margin', 'restored_margin', 'patch_effect',
                             'remaining_effect', 'mediated_effect', 'reference_source_margin']
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for index, row in enumerate(rows):
            prefix = prefixes[index].unsqueeze(0).to(device)
            mask = torch.ones(prefix.shape[:2], device=device, dtype=torch.bool)
            clean_cache = {}

            def forward(audio=None, query_group=None, capture=False, source_reference=False):
                handles = []
                current_cache = {}
                def capture_v(layer):
                    def hook(_module, _inputs, output):
                        clean_cache[layer] = output.detach().cpu().clone()
                    return hook
                def capture_inputs(layer):
                    def hook(_module, _args, kwargs):
                        if kwargs.get('past_key_values') is not None:
                            raise ValueError('Edge restore requires cache-free forward')
                        current_cache[layer] = dict(position_embeddings=kwargs['position_embeddings'],
                                                    attention_mask=kwargs.get('attention_mask'))
                    return hook
                def capture_projection(layer, name):
                    def hook(_module, _inputs, output):
                        current_cache[layer][name] = output
                    return hook
                def restore_edge(layer):
                    def hook(module, inputs):
                        state = current_cache[layer]
                        attn = layers[layer].self_attn
                        if attn.training or attn.sliding_window is not None:
                            raise ValueError('Requires eval attention without sliding window')
                        def reshape(value):
                            return value.view(value.shape[0], value.shape[1], -1, attn.head_dim).transpose(1, 2)
                        q, k = apply_rotary_pos_emb(reshape(state['q']), reshape(state['k']), *state['position_embeddings'])
                        k = repeat_kv(k, attn.num_key_value_groups)
                        v = repeat_kv(reshape(state['v']), attn.num_key_value_groups)
                        clean_v = repeat_kv(reshape(clean_cache[layer].to(device=v.device, dtype=v.dtype)), attn.num_key_value_groups)
                        queries = groups[query_group]
                        original = inputs[0]
                        reconstructed = selected_attention(torch, q, k, v, queries, state['attention_mask'], attn.scaling)
                        error = float((reconstructed-original[:, queries, :]).abs().max())
                        if not math.isfinite(error):
                            raise RuntimeError('Non-finite attention reconstruction')
                        checks['attention_reconstruction_max'] = max(checks['attention_reconstruction_max'], error)
                        restored = edge_output(torch, original, q, k, v, clean_v, sources, queries, state['attention_mask'], attn.scaling)
                        early = [query for query in queries if query < positions['audio_start']]
                        if early:
                            early_error = float((restored[:, early, :]-original[:, early, :]).abs().max())
                            if not math.isfinite(early_error):
                                raise RuntimeError('Non-finite preaudio query correction')
                            checks['pre_audio_query_delta_max'] = max(checks['pre_audio_query_delta_max'], early_error)
                        others = sorted(set(range(original.shape[1])) - set(queries))
                        if others:
                            checks['nonselected_rows_max'] = max(checks['nonselected_rows_max'], float((restored[:, others, :]-original[:, others, :]).abs().max()))
                        return (restored,) + inputs[1:]
                    return hook
                try:
                    if audio is not None:
                        handles.append(layers[17].register_forward_hook(base._patch_hook(audio, 'audio_tokens', positions['audio_start'], positions['audio_length'], positions['decision_position'])))
                    for layer in base.RESTORE_LAYERS:
                        attn = layers[layer].self_attn
                        if capture:
                            handles.append(attn.v_proj.register_forward_hook(capture_v(layer)))
                        if source_reference:
                            handles.append(attn.v_proj.register_forward_hook(base.source_value_restore(clean_cache[layer], sources)))
                        if query_group is not None:
                            handles.append(attn.register_forward_pre_hook(capture_inputs(layer), with_kwargs=True))
                            for name in ('q', 'k', 'v'):
                                handles.append(getattr(attn, name+'_proj').register_forward_hook(capture_projection(layer, name)))
                            handles.append(attn.o_proj.register_forward_pre_hook(restore_edge(layer)))
                    with torch.inference_mode():
                        result = base._decoder_forward(model, prefix, mask)
                        logits = model.llm.lm_head(result.last_hidden_state[:, positions['decision_position'], :]).float()
                        value = float(logits[0, 6247]-logits[0, 12421])
                        if not math.isfinite(value):
                            raise RuntimeError('Non-finite margin')
                        return value
                finally:
                    for handle in handles:
                        handle.remove()

            clean = forward(capture=True)
            donor = donors[counterpart[index]:counterpart[index]+1]
            patched = forward(audio=donor)
            reference = forward(audio=donor, source_reference=True)
            for name, audio, expected in [('clean', None, clean), ('patched', donor, patched)]:
                kwargs = {} if audio is None else dict(layer_module=layers[17], donor=audio, patch_kind='audio_tokens', **{k: positions[k] for k in ('audio_start', 'audio_length', 'decision_position')})
                happy, sad = base._score_candidate_pairs(torch, model, prefix, mask, ids[0], ids[1], positions['prefix_length'], **kwargs)
                error = abs(float(happy[0]-sad[0])-expected)
                if not math.isfinite(error):
                    raise RuntimeError('Non-finite candidate parity')
                checks[name+'_candidate_parity_max'] = max(checks[name+'_candidate_parity_max'], error)
            if index == 0:
                checks['self_audio_max'] = abs(forward(audio=donors[index:index+1])-clean)
            sign = 1 if row['emotion'] == 'sad' else -1
            common = {key: row[key] for key in metadata_keys}
            common.update(donor_sample_id=rows[counterpart[index]]['sample_id'], donor_sign=sign)
            for name in groups:
                restored = forward(audio=donor, query_group=name)
                if index == 0:
                    checks['component_noop_max'] = max(checks['component_noop_max'], abs(forward(query_group=name)-clean))
                if name == 'all':
                    checks['all_query_source_reference_max'] = max(checks['all_query_source_reference_max'], abs(restored-reference))
                writer.writerow(dict(common, condition=f'edge_{name}_joint', component='audio_v_edge', restore_layers=json.dumps(base.RESTORE_LAYERS), clean_margin=clean, patched_margin=patched, restored_margin=restored, patch_effect=sign*(patched-clean), remaining_effect=sign*(restored-clean), mediated_effect=sign*(patched-restored), reference_source_margin=reference))
            stream.flush()
            if any(not math.isfinite(value) or value > args.self_patch_tolerance for value in checks.values()):
                raise RuntimeError(f'Numerical validation failed: {checks}')
            LOGGER.info('completed sample %d/%d checks=%s', index+1, len(rows), checks)
    meta = dict(n_samples=len(rows), n_pairs=len(pairs), seed=args.seed, positions=positions,
                source_positions=sources, query_positions=groups,
                source_spans_half_open=base._spans(sources), query_spans_half_open={key: base._spans(value) for key, value in groups.items()},
                condition_id='upper_prompt__lower_spaced', prompt=condition['prompt'], happy_token_ids=ids[0], sad_token_ids=ids[1],
                checkpoint=str(args.checkpoint), model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                torch_version=torch.__version__, transformers_version=transformers.__version__,
                attention_implementation=model.llm.config._attn_implementation, model_dtype=str(next(model.parameters()).dtype),
                numerical_checks=checks, tolerance=args.self_patch_tolerance, candidate_parity_samples=len(rows), source_reference_samples=len(rows), noop_samples=1,
                upstream_patch='opposite emotion donor audio at post-block 17', restore_layers=base.RESTORE_LAYERS,
                restore_scope='pre-o_proj selected query rows += current attention weights @ audio-only (clean-target V minus current V); QK and other query rows unchanged within each hook',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(base.parse_args())
