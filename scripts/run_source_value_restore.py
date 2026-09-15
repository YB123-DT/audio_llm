#!/usr/bin/env python3
"""Restore clean target V at selected source positions after layer17 audio swap.

Projection hooks affect every query attending those sources, not only decision.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
try:
    from scripts.run_residual_component_restore import (
        DEFAULT_SEED, FORCED_CHOICE_CONDITIONS, _capture_all_donors,
        _decoder_forward, _hidden_tensor, _llm_layers, _pair_indices,
        _patch_hook, _prepare_prefixes, _replace_hidden, _score_candidate_pairs,
        load_model, read_rows,
    )
except ModuleNotFoundError:
    from run_residual_component_restore import (
        DEFAULT_SEED, FORCED_CHOICE_CONDITIONS, _capture_all_donors,
        _decoder_forward, _hidden_tensor, _llm_layers, _pair_indices,
        _patch_hook, _prepare_prefixes, _replace_hidden, _score_candidate_pairs,
        load_model, read_rows,
    )

LOGGER = logging.getLogger(__name__)
RESTORE_LAYERS = list(range(18, 24))


def source_partitions(positions):
    start, length = positions['audio_start'], positions['audio_length']
    total, decision = positions['prefix_length'], positions['decision_position']
    end = start + length
    if not (start >= 2 and length > 0 and end + 2 == total and decision == total - 1):
        raise ValueError('Expected prompt, input marker, audio, end marker, decision layout')
    groups = dict(audio=list(range(start, end)), prompt=list(range(start - 1)),
                  other=[start - 1, end, decision])
    if sorted(groups['audio'] + groups['prompt'] + groups['other']) != list(range(total)):
        raise ValueError('Audio/prompt/other must be a disjoint exhaustive partition')
    groups.update(non_audio=groups['prompt'] + groups['other'], decision=[decision],
                  post_audio_marker=[end], all=list(range(total)))
    return groups


def source_value_restore(clean, indices):
    """Replace selected V source rows, preserving every other row and input."""
    def hook(_module, _inputs, output):
        hidden = _hidden_tensor(output)
        if clean.shape != hidden.shape:
            raise ValueError('Source restore requires identical prefix shapes')
        result = hidden.clone()
        result[:, indices, :] = clean[:, indices, :].to(hidden.device, hidden.dtype)
        return _replace_hidden(output, result)
    return hook


def run(args):
    torch, model, tokenizer, whisper = load_model(args)
    import transformers
    model.requires_grad_(False)
    rows = read_rows(args.manifest, args.limit)[args.start:]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError('No complete matched pairs')
    layers = _llm_layers(model)
    if len(layers) != 24:
        raise ValueError('Expected 24 decoder blocks')
    condition = FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    ids = [tokenizer.encode(condition[k], add_special_tokens=False)
           for k in ('happy_verbalizer', 'sad_verbalizer')]
    if ids != [[6247], [12421]]:
        raise ValueError(f'Unexpected single-token verbalizers: {ids}')
    prefixes, positions = _prepare_prefixes(torch, model, whisper, tokenizer, rows,
                                            args.ravdess_root, condition['prompt'])
    groups = source_partitions(positions)
    device = next(model.parameters()).device
    donors = _capture_all_donors(torch, model, prefixes, positions, (17,), device,
                                args.batch_size)[17]['audio_tokens']
    counterpart = {}
    for pair in pairs:
        h, s = pair['happy_index'], pair['sad_index']
        counterpart[h], counterpart[s] = s, h
    if len(counterpart) != len(rows):
        raise ValueError('Every sample must have exactly one matched counterpart')
    checks = dict(clean_candidate_parity_max=0.0, patched_candidate_parity_max=0.0,
                  self_audio_max=0.0, component_noop_max=0.0,
                  pre_audio_prompt_v_max=0.0)
    metadata_keys = ['sample_id', 'pair_id', 'actor', 'statement', 'repetition', 'intensity', 'emotion']
    fields = metadata_keys + ['donor_sample_id', 'donor_sign', 'condition', 'component',
        'restore_layers', 'clean_margin', 'patched_margin', 'restored_margin',
        'patch_effect', 'remaining_effect', 'mediated_effect']
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for index, row in enumerate(rows):
            prefix = prefixes[index].unsqueeze(0).to(device)
            mask = torch.ones(prefix.shape[:2], device=device, dtype=torch.bool)
            cache = {}

            def forward(audio=None, source=None, capture=False, check_prompt=False):
                handles = []
                def capture_v(layer):
                    def hook(_module, _inputs, output):
                        cache[layer] = _hidden_tensor(output).detach().cpu().clone()
                    return hook
                def compare_prompt(layer):
                    def hook(_module, _inputs, output):
                        observed = _hidden_tensor(output)[:, list(range(positions['audio_start'])), :].detach().cpu()
                        error = float((observed - cache[layer][:, :positions['audio_start'], :]).abs().max())
                        if not math.isfinite(error):
                            raise RuntimeError('Non-finite preaudio V difference')
                        checks['pre_audio_prompt_v_max'] = max(checks['pre_audio_prompt_v_max'], error)
                    return hook
                try:
                    if audio is not None:
                        handles.append(layers[17].register_forward_hook(_patch_hook(
                            audio, 'audio_tokens', positions['audio_start'], positions['audio_length'],
                            positions['decision_position'])))
                    for layer in RESTORE_LAYERS:
                        module = layers[layer].self_attn.v_proj
                        if capture:
                            handles.append(module.register_forward_hook(capture_v(layer)))
                        if check_prompt:
                            handles.append(module.register_forward_hook(compare_prompt(layer)))
                        if source is not None:
                            handles.append(module.register_forward_hook(source_value_restore(cache[layer], groups[source])))
                    with torch.inference_mode():
                        result = _decoder_forward(model, prefix, mask)
                        logits = model.llm.lm_head(result.last_hidden_state[:, positions['decision_position'], :]).float()
                        margin = float(logits[0, 6247] - logits[0, 12421])
                        if not math.isfinite(margin):
                            raise RuntimeError('Non-finite margin')
                        return margin
                finally:
                    for handle in handles:
                        handle.remove()

            clean = forward(capture=True)
            donor = donors[counterpart[index]:counterpart[index] + 1]
            patched = forward(audio=donor, check_prompt=True)
            for name, audio, expected in [('clean', None, clean), ('patched', donor, patched)]:
                kwargs = {} if audio is None else dict(layer_module=layers[17], donor=audio,
                    patch_kind='audio_tokens', **{k: positions[k] for k in
                    ('audio_start', 'audio_length', 'decision_position')})
                happy, sad = _score_candidate_pairs(torch, model, prefix, mask, ids[0], ids[1],
                                                    positions['prefix_length'], **kwargs)
                key = name + '_candidate_parity_max'
                error = abs(float(happy[0] - sad[0]) - expected)
                if not math.isfinite(error):
                    raise RuntimeError('Non-finite candidate parity difference')
                checks[key] = max(checks[key], error)
            if index == 0:
                checks['self_audio_max'] = abs(forward(audio=donors[index:index+1]) - clean)
            sign = 1 if row['emotion'] == 'sad' else -1
            common = {key: row[key] for key in metadata_keys}
            common.update(donor_sample_id=rows[counterpart[index]]['sample_id'], donor_sign=sign)
            for source in groups:
                restored = forward(audio=donor, source=source)
                if index == 0:
                    checks['component_noop_max'] = max(checks['component_noop_max'], abs(forward(source=source)-clean))
                writer.writerow(dict(common, condition=f'v_{source}_joint', component='v',
                    restore_layers=json.dumps(RESTORE_LAYERS), clean_margin=clean,
                    patched_margin=patched, restored_margin=restored,
                    patch_effect=sign*(patched-clean), remaining_effect=sign*(restored-clean),
                    mediated_effect=sign*(patched-restored)))
            output.flush()
            if checks['pre_audio_prompt_v_max'] != 0 or any(v > args.self_patch_tolerance for v in checks.values()):
                raise RuntimeError(f'Numerical validation failed: {checks}')
            LOGGER.info('completed sample %d/%d checks=%s', index+1, len(rows), checks)
    meta = dict(n_samples=len(rows), n_pairs=len(pairs), seed=args.seed, positions=positions,
        source_positions=groups, source_spans_half_open={key: _spans(value) for key, value in groups.items()},
        token_roles={'prompt': 'text prompt including its input_t and eot markers',
                     'pre_audio_input_marker': positions['audio_start']-1,
                     'post_audio_eoa_eot_marker': positions['audio_start']+positions['audio_length'],
                     'decision_answer_a_answer_t': positions['decision_position']},
        condition_id='upper_prompt__lower_spaced', prompt=condition['prompt'],
        happy_token_ids=ids[0], sad_token_ids=ids[1], checkpoint=str(args.checkpoint),
        model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        attention_implementation=model.llm.config._attn_implementation,
        model_dtype=str(next(model.parameters()).dtype), numerical_checks=checks,
        tolerance=args.self_patch_tolerance, candidate_parity_samples=len(rows), noop_samples=1,
        upstream_patch='opposite emotion donor audio at post-block 17', restore_layers=RESTORE_LAYERS,
        restore_scope='clean target v_proj outputs at selected sources for ALL queries; not a decision-edge-specific intervention',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(meta, indent=2)+'\n')


def _spans(indices):
    spans = []
    for index in sorted(indices):
        if spans and spans[-1][1] == index:
            spans[-1][1] += 1
        else:
            spans.append([index, index+1])
    return spans


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'ravdess-root', 'slam-llm-root', 'qwen-path', 'whisper-path', 'checkpoint', 'output-csv'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--self-patch-tolerance', type=float, default=1e-3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Positive batch size required')
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(parse_args())
