#!/usr/bin/env python3
"""Fixed CREMA-D confirmation: early answer-only Attention versus MLP updates."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import logging
from pathlib import Path
import numpy as np
try:
    from . import run_activation_patching as base
    from .run_early_answer_ablation import CONDITIONS, TARGET_BLOCKS, install_hooks
except ImportError:
    import run_activation_patching as base
    from run_early_answer_ablation import CONDITIONS, TARGET_BLOCKS, install_hooks


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_rows(rows):
    """Require complete matched pairs, never exclude samples based on outcomes."""
    ids = [row['sample_id'] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Manifest must have nonempty unique sample IDs')
    pairs = {}
    for row in rows:
        pairs.setdefault(row['pair_id'], []).append(row)
    for pair in pairs.values():
        if len(pair) != 2 or {r['emotion'] for r in pair} != {'happy', 'sad'}:
            raise ValueError('Every pair must contain exactly happy and sad')
        for key in ('actor', 'statement', 'intensity', 'repetition'):
            if len({r[key] for r in pair}) != 1:
                raise ValueError(f'Matched pair differs in {key}')
    return ids


def validate_audio_checksums(rows, checksum_path, audio_root):
    """Bind every selected decoded recording to the frozen extraction inputs."""
    with checksum_path.open(newline='') as stream:
        checksums = list(csv.DictReader(stream))
    indexed = {row['sample_id']: row for row in checksums}
    if len(indexed) != len(checksums) or set(indexed) != {r['sample_id'] for r in rows}:
        raise ValueError('Audio checksum coverage differs from manifest')
    for row in rows:
        entry = indexed[row['sample_id']]
        if entry['path'] != row['path']:
            raise ValueError('Audio checksum path differs from manifest')
        if int(entry['sample_rate']) <= 0 or int(entry['frames']) <= 0 or not np.isfinite(float(entry['duration_seconds'])) or float(entry['duration_seconds']) <= 0:
            raise ValueError('Invalid decoded audio metadata')
        if sha256(audio_root / row['path']) != entry['sha256']:
            raise ValueError('Audio checksum mismatch: ' + row['sample_id'])


def save_npz(path, arrays):
    """Atomic checkpoint: an interrupted write cannot replace the last complete one."""
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def validate_resume(saved, ids, fingerprint):
    if str(saved['fingerprint'].item()) != fingerprint:
        raise ValueError('Resume provenance differs from this run')
    done = len(saved['sample_ids'])
    if done > len(ids) or list(saved['sample_ids']) != ids[:done]:
        raise ValueError('Resume sample order differs')
    if list(saved['conditions']) != list(CONDITIONS):
        raise ValueError('Resume conditions differ')
    for key in ('block6', 'final_answer'):
        if saved[key].shape != (done, 3, 896) or not np.isfinite(saved[key]).all():
            raise ValueError(f'Invalid resume {key}')
    if saved['clean_margins'].shape != (done,) or not np.isfinite(saved['clean_margins']).all():
        raise ValueError('Invalid resume margins')
    checks = json.loads(str(saved['checks'].item()))
    if checks['nonanswer_block_comparisons'] != done * 48 or checks['suppressed_update_calls'] != done * 12:
        raise ValueError('Resume validation counts differ')
    if done and checks.get('unhooked_clean_final_max') != 0:
        raise ValueError('Missing clean unhooked parity check')
    return done, checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'audio-checksums', 'audio-root', 'slam-llm-root', 'qwen-path', 'whisper-path', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-every', type=int, default=24)
    args = p.parse_args()
    if args.checkpoint_every < 1:
        p.error('--checkpoint-every must be positive')
    args.seed = 1234
    rows = base.read_rows(args.manifest, None)
    ids = validate_rows(rows)
    if sha256(args.manifest) != 'b7db3dff0af5a0e18a435ebbe888181d001501a190f4bba9f937ed2c8c6993a3':
        raise ValueError('Manifest differs from the frozen CREMA-D cohort')
    validate_audio_checksums(rows, args.audio_checksums, args.audio_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resume_path = args.output.with_suffix('.resume.npz')
    files = [Path(__file__), Path(base.__file__), Path(inspect.getfile(install_hooks)),
             Path(inspect.getfile(base.load_model)), args.manifest, args.audio_checksums, args.whisper_path, args.checkpoint]
    files.extend(sorted(p for p in args.qwen_path.iterdir() if p.is_file() and (p.suffix == '.json' or p.name in ('merges.txt', 'tokenizer.model'))))
    hashes = {str(f): sha256(f) for f in files}
    torch, model, tokenizer, whisper = base.load_model(args)
    model.requires_grad_(False)
    blocks = base._llm_layers(model)
    if len(blocks) != 24:
        raise ValueError('Expected 24 blocks')
    import transformers
    source_path = Path(inspect.getfile(type(blocks[0])))
    hashes[str(source_path)] = sha256(source_path)
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, dtype=str(next(model.parameters()).dtype), seed=1234)
    fingerprint = hashlib.sha256(json.dumps(dict(files=hashes, runtime=runtime), sort_keys=True).encode()).hexdigest()
    block6, finals, margins = [], [], []
    checks = {'nonanswer_block_comparisons': 0, 'suppressed_update_calls': 0}
    positions = None
    if args.resume and resume_path.exists():
        with np.load(resume_path, allow_pickle=False) as saved:
            done, checks = validate_resume(saved, ids, fingerprint)
            block6, finals, margins = list(saved['block6']), list(saved['final_answer']), list(saved['clean_margins'])
            positions = json.loads(str(saved['positions'].item()))
        logging.info('Resuming after %d complete samples', done)
    elif resume_path.exists() or args.output.exists():
        raise FileExistsError('Output/checkpoint exists; use --resume or a new output path')
    verbalizer = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    if [tokenizer.encode(verbalizer[k], add_special_tokens=False) for k in ('happy_verbalizer', 'sad_verbalizer')] != [[6247], [12421]]:
        raise ValueError('Verbalizer token mismatch')
    device = next(model.parameters()).device
    with torch.inference_mode():
        d_lm = (model.llm.lm_head.weight[6247] - model.llm.lm_head.weight[12421]).float().cpu().numpy()
        for i in range(len(block6), len(rows)):
            prefixes, current = base._prepare_prefixes(torch, model, whisper, tokenizer, [rows[i]], args.audio_root, verbalizer['prompt'])
            if current != {'audio_start': 29, 'audio_length': 300, 'prefix_length': 331, 'decision_position': 330}:
                raise ValueError(f'Unexpected fixed prefix layout: {current}')
            if positions is not None and current != positions:
                raise ValueError('Prefix layout changed')
            positions = current
            decision = positions['decision_position']
            prefix = prefixes[0].unsqueeze(0).to(device)
            mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=device)
            unhooked = None
            if i == 0:
                unhooked = base._decoder_forward(model, prefix, mask).last_hidden_state[0, decision].detach().clone()
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
                        if unhooked is not None:
                            checks['unhooked_clean_final_max'] = float((final - unhooked).abs().max())
                            if not final.equal(unhooked):
                                raise AssertionError('Clean observer hooks changed the final state')
                finally:
                    for handle in handles:
                        handle.remove()
                if calls != ([] if condition == 'clean' else list(TARGET_BLOCKS)):
                    raise AssertionError(f'Wrong intervention calls: {calls}')
                checks['suppressed_update_calls'] += len(calls)
            block6.append(np.stack(sample_b6))
            finals.append(np.stack(sample_final))
            if (i + 1) % args.checkpoint_every == 0 or i + 1 == len(rows):
                save_npz(resume_path, dict(sample_ids=np.array(ids[:i+1]), conditions=np.array(CONDITIONS), block6=np.stack(block6), final_answer=np.stack(finals), clean_margins=np.array(margins), fingerprint=np.array(fingerprint), checks=np.array(json.dumps(checks)), positions=np.array(json.dumps(positions))))
            logging.info('Completed sample %d/%d (three conditions)', i + 1, len(rows))
    arrays = dict(sample_ids=np.array(ids), conditions=np.array(CONDITIONS), block6=np.stack(block6), final_answer=np.stack(finals), clean_margins=np.array(margins), d_lm=d_lm)
    if checks['nonanswer_block_comparisons'] != len(ids)*48 or checks['suppressed_update_calls'] != len(ids)*12:
        raise AssertionError('Wrong validation count')
    for key, value in arrays.items():
        if key not in ('sample_ids', 'conditions') and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite {key}')
    save_npz(args.output, arrays)
    provenance = dict(dataset='CREMA-D', seed=1234, n_samples=len(ids), conditions=CONDITIONS, target_blocks=TARGET_BLOCKS, positions=positions, checks=checks, shapes={k:list(v.shape) for k,v in arrays.items()}, prompt=verbalizer['prompt'], happy_token_id=6247, sad_token_id=12421, decoder_forwards_per_sample=3, additional_unhooked_validation_forwards=1, model_parameters_frozen=all(not param.requires_grad for param in model.parameters()), torch_version=torch.__version__, transformers_version=transformers.__version__, dtype=str(next(model.parameters()).dtype), checkpoint=str(args.checkpoint), state_semantics={'block6':'Complete answer post-block6 residual', 'final_answer':'Complete final answer after terminal RMSNorm'}, intervention='Zero answer row of self_attn or mlp update before residual addition at zero-based blocks1..6. Every non-answer row at every block must remain bitwise equal to clean.', decoder_forward_source=inspect.getsource(type(blocks[0]).forward), sha256=hashes, output_sha256=sha256(args.output), resume_fingerprint=fingerprint, runtime_fingerprint_fields=runtime, audio_checksums_verified=len(ids))
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
