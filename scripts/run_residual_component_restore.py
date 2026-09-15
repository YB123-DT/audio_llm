#!/usr/bin/env python3
"""Frozen, prefix-only residual attribution and component restoration.

Restore component contributions, never the residual sum. QK/V restores are
all-source interventions and are disabled unless --qkv-layers is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

try:
    from scripts.run_activation_patching import (
        DEFAULT_SEED, FORCED_CHOICE_CONDITIONS, _capture_all_donors,
        _decoder_forward, _hidden_tensor, _llm_layers, _pair_indices,
        _patch_hook, _prepare_prefixes, _replace_hidden, _score_candidate_pairs,
        load_model, read_rows,
    )
except ModuleNotFoundError:
    from run_activation_patching import (
        DEFAULT_SEED, FORCED_CHOICE_CONDITIONS, _capture_all_donors,
        _decoder_forward, _hidden_tensor, _llm_layers, _pair_indices,
        _patch_hook, _prepare_prefixes, _replace_hidden, _score_candidate_pairs,
        load_model, read_rows,
    )

LOGGER = logging.getLogger(__name__)


def contribution_restore(clean, decision_position, all_positions=False):
    """Replace a module's contribution before residual addition."""
    def hook(_module, _inputs, output):
        hidden = _hidden_tensor(output)
        result = hidden.clone()
        replacement = clean.to(device=hidden.device, dtype=hidden.dtype)
        if all_positions:
            if replacement.shape != result.shape:
                raise ValueError('All-source restore requires identical prefix shapes')
            result.copy_(replacement)
        else:
            result[:, decision_position, :] = replacement
        return _replace_hidden(output, result)
    return hook


def condition_specs(qkv_layers):
    specs = [(f'{kind}_{layer}', kind, [layer]) for layer in range(18, 24)
             for kind in ('attention', 'mlp')]
    specs += [(f'{kind}_joint', kind, list(range(18, 24))) for kind in ('attention', 'mlp')]
    if qkv_layers:
        specs += [(f'{kind}_{layer}', kind, [layer]) for layer in qkv_layers for kind in ('qk', 'v')]
        specs += [(f'{kind}_joint', kind, list(qkv_layers)) for kind in ('qk', 'v')]
        specs += [('qkv_joint', 'qkv', list(qkv_layers))]
    return specs


def run(args):
    torch, model, tokenizer, whisper = load_model(args)
    model.requires_grad_(False)
    import transformers
    rows = read_rows(args.manifest, args.limit)[args.start:]
    pairs = _pair_indices(rows)
    if not pairs:
        raise ValueError('No complete matched pairs')
    layers = _llm_layers(model)
    if len(layers) != 24:
        raise ValueError('This preregistered runner requires 24 decoder blocks')
    condition = FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    ids = [tokenizer.encode(condition[k], add_special_tokens=False) for k in ('happy_verbalizer', 'sad_verbalizer')]
    if ids != [[6247], [12421]]:
        raise ValueError(f'Unexpected single-token verbalizers: {ids}')
    prefixes, positions = _prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    device = next(model.parameters()).device
    donors = _capture_all_donors(torch, model, prefixes, positions, (17,), device, args.batch_size)[17]['audio_tokens']
    counterpart = {}
    for pair in pairs:
        h, s = pair['happy_index'], pair['sad_index']
        counterpart[h], counterpart[s] = s, h
    if len(counterpart) != len(rows):
        raise ValueError('Every sample must belong to exactly one complete pair')
    decision = positions['decision_position']
    norm = model.llm.model.norm
    head = model.llm.lm_head
    direction = (head.weight[6247].float() - head.weight[12421].float()).detach()
    bias = 0.0 if head.bias is None else float((head.bias[6247] - head.bias[12421]).float())
    specs = condition_specs(args.qkv_layers)
    checks = {'clean_candidate_parity_max': 0.0, 'patched_candidate_parity_max': 0.0,
              'final_lens_parity_max': 0.0, 'self_audio_max': 0.0, 'component_noop_max': 0.0, 'pre_routing_decision_max': 0.0}
    metadata_keys = ['sample_id', 'pair_id', 'actor', 'statement', 'repetition', 'intensity', 'emotion']
    shared = metadata_keys + ['donor_sample_id', 'donor_sign']
    out_fields = shared + ['condition', 'component', 'restore_layers', 'clean_margin', 'patched_margin', 'restored_margin', 'patch_effect', 'remaining_effect', 'mediated_effect']
    attr_fields = shared + ['layer_index', 'clean_raw', 'patched_raw', 'delta_raw', 'clean_lens', 'patched_lens', 'delta_lens']
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    attr_path = args.output_csv.with_name(args.output_csv.stem + '_attribution.csv')
    with args.output_csv.open('w', newline='') as out, attr_path.open('w', newline='') as attr:
        writer = csv.DictWriter(out, fieldnames=out_fields, lineterminator="\n"); writer.writeheader()
        awriter = csv.DictWriter(attr, fieldnames=attr_fields, lineterminator="\n"); awriter.writeheader()
        for index, row in enumerate(rows):
            prefix = prefixes[index].unsqueeze(0).to(device)
            mask = torch.ones(prefix.shape[:2], device=device, dtype=torch.bool)
            clean_cache = {}
            attrs = {}

            def forward(audio=None, restore=None, capture=False, capture_attrs=False):
                handles = []
                current_attrs = {}
                def capture_component(key, all_positions=False):
                    def hook(_module, _inputs, output):
                        h = _hidden_tensor(output)
                        clean_cache[key] = (h if all_positions else h[:, decision, :]).detach().cpu().clone()
                    return hook
                def capture_residual(layer):
                    def hook(_module, _inputs, output):
                        h = _hidden_tensor(output)[:, decision, :]
                        raw = float((h.float() * direction).sum())
                        lens = float((norm(h).float() * direction).sum()) + bias
                        current_attrs[layer] = (raw, lens)
                    return hook
                try:
                    if audio is not None:
                        handles.append(layers[17].register_forward_hook(_patch_hook(audio, 'audio_tokens', positions['audio_start'], positions['audio_length'], decision)))
                    for layer in range(24):
                        block = layers[layer]
                        modules = [('attention', block.self_attn, False), ('mlp', block.mlp, False)] if layer >= 18 else []
                        if layer in args.qkv_layers:
                            modules += [('q', block.self_attn.q_proj, False), ('k', block.self_attn.k_proj, True), ('v', block.self_attn.v_proj, True)]
                        for kind, module, all_positions in modules:
                            if capture:
                                handles.append(module.register_forward_hook(capture_component((layer, kind), all_positions)))
                            if restore is not None:
                                restore_kind, selected = restore
                                kinds = (('q', 'k', 'v') if restore_kind == 'qkv' else
                                         ('q', 'k') if restore_kind == 'qk' else (restore_kind,))
                                if layer in selected and kind in kinds:
                                    handles.append(module.register_forward_hook(contribution_restore(clean_cache[layer, kind], decision, all_positions)))
                        if capture_attrs:
                            handles.append(block.register_forward_hook(capture_residual(layer)))
                    with torch.inference_mode():
                        result = _decoder_forward(model, prefix, mask)
                        logits = head(result.last_hidden_state[:, decision, :]).float()
                        margin = float(logits[0, 6247] - logits[0, 12421])
                    if capture_attrs:
                        checks['final_lens_parity_max'] = max(checks['final_lens_parity_max'], abs(current_attrs[23][1] - margin))
                    return margin, current_attrs
                finally:
                    for handle in handles:
                        handle.remove()

            clean, attrs['clean'] = forward(capture=True, capture_attrs=True)
            donor = donors[counterpart[index]:counterpart[index] + 1]
            patched, attrs['patched'] = forward(audio=donor, capture_attrs=True)
            checks['pre_routing_decision_max'] = max(checks['pre_routing_decision_max'], max(abs(attrs['patched'][l][j]-attrs['clean'][l][j]) for l in range(18) for j in range(2)))
            # Verify equivalence for every sample, with and without upstream patch.
            for name, audio, expected in [('clean', None, clean), ('patched', donor, patched)]:
                kwargs = {} if audio is None else dict(layer_module=layers[17], donor=audio, patch_kind='audio_tokens', **{k: positions[k] for k in ('audio_start', 'audio_length', 'decision_position')})
                happy, sad = _score_candidate_pairs(torch, model, prefix, mask, ids[0], ids[1], positions['prefix_length'], **kwargs)
                error = abs(float(happy[0] - sad[0]) - expected)
                checks[name + '_candidate_parity_max'] = max(checks[name + '_candidate_parity_max'], error)
            sign = 1 if row['emotion'] == 'sad' else -1
            common = {k: row[k] for k in metadata_keys}
            common.update(donor_sample_id=rows[counterpart[index]]['sample_id'], donor_sign=sign)
            for layer in range(24):
                cr, cl = attrs['clean'][layer]; pr, pl = attrs['patched'][layer]
                awriter.writerow(dict(common, layer_index=layer, clean_raw=cr, patched_raw=pr, delta_raw=sign*(pr-cr), clean_lens=cl, patched_lens=pl, delta_lens=sign*(pl-cl)))
            if index == 0:
                own, _ = forward(audio=donors[index:index+1])
                checks['self_audio_max'] = abs(own-clean)
            for label, component, selected in specs:
                restored, _ = forward(audio=donor, restore=(component, selected))
                if index == 0:
                    noop, _ = forward(restore=(component, selected))
                    checks['component_noop_max'] = max(checks['component_noop_max'], abs(noop-clean))
                writer.writerow(dict(common, condition=label, component=component, restore_layers=json.dumps(selected), clean_margin=clean, patched_margin=patched, restored_margin=restored, patch_effect=sign*(patched-clean), remaining_effect=sign*(restored-clean), mediated_effect=sign*(patched-restored)))
            out.flush(); attr.flush()
            if any(v > args.self_patch_tolerance for v in checks.values()):
                raise RuntimeError(f'Numerical validation failed: {checks}')
            LOGGER.info('completed sample %d/%d checks=%s', index+1, len(rows), checks)
    meta = dict(n_samples=len(rows), n_pairs=len(pairs), seed=args.seed, positions=positions,
                condition_id='upper_prompt__lower_spaced', prompt=condition['prompt'], happy_token_ids=ids[0], sad_token_ids=ids[1],
                checkpoint=str(args.checkpoint), model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                torch_version=torch.__version__, transformers_version=transformers.__version__,
                attention_implementation=model.llm.config._attn_implementation,
                model_dtype=str(next(model.parameters()).dtype), numerical_checks=checks, tolerance=args.self_patch_tolerance,
                candidate_parity_samples=len(rows), noop_samples=1,
                upstream_patch='opposite emotion donor audio at post-block 17',
                component_restore='clean target module contribution at decision only, before residual addition',
                qkv_layers=args.qkv_layers, qkv_scope='Q decision and K all prefix sources; V all prefix sources; projected states before RoPE',
                attribution='post-block decision residual; raw dot W_happy-W_sad and final-RMSNorm logit lens; delta is donor oriented',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(meta, indent=2)+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'ravdess-root', 'slam-llm-root', 'qwen-path', 'whisper-path', 'checkpoint', 'output-csv'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='Donor capture batch; interventions use one sample at a time')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--self-patch-tolerance', type=float, default=1e-3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--qkv-layers', type=int, nargs='+', default=[])
    args = parser.parse_args()
    if args.batch_size < 1 or any(layer not in range(18,24) for layer in args.qkv_layers):
        parser.error('Positive batch size and QKV layers 18..23 required')
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(parse_args())
