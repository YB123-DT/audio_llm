#!/usr/bin/env python3
"""Fixed LLaMA-Omni2 answer-update replication on the two frozen cohorts."""
import argparse
import inspect
import json
import logging
from pathlib import Path
import random
import numpy as np
try:
    from .llama_omni2_runtime import load_runtime, prepare_prefix, validate_forward_parity, sha256
    from .run_early_answer_ablation import CONDITIONS, TARGET_BLOCKS, install_hooks
    from .run_crema_confirmation import validate_rows, validate_audio_checksums, save_npz
    from .slam_omni_diagnostics import read_rows
except ImportError:
    from llama_omni2_runtime import load_runtime, prepare_prefix, validate_forward_parity, sha256
    from run_early_answer_ablation import CONDITIONS, TARGET_BLOCKS, install_hooks
    from run_crema_confirmation import validate_rows, validate_audio_checksums, save_npz
    from slam_omni_diagnostics import read_rows

COHORTS = {'ravdess': (192, 'c1894d6dc765947521254a62ef7cc582519b0110acc52e6a6225be9806bf74d6'),
           'crema-d': (1936, 'b7db3dff0af5a0e18a435ebbe888181d001501a190f4bba9f937ed2c8c6993a3')}


def validate_cohort(dataset, manifest, rows):
    ids = validate_rows(rows)
    count, expected_hash = COHORTS[dataset]
    if len(ids) != count or sha256(manifest) != expected_hash:
        raise ValueError('Dataset differs from frozen cohort')
    return ids


def validate_replication_resume(saved, ids, fingerprint):
    if str(saved['fingerprint'].item()) != fingerprint:
        raise ValueError('Resume provenance mismatch')
    done = len(saved['sample_ids'])
    if not 0 < done <= len(ids) or saved['sample_ids'].tolist() != ids[:done] or tuple(saved['conditions']) != CONDITIONS:
        raise ValueError('Resume identity or condition mismatch')
    for endpoint in ('block6', 'final_answer'):
        if saved[endpoint].shape != (done, 3, 896) or not np.isfinite(saved[endpoint]).all():
            raise ValueError('Invalid resume states')
    checks = json.loads(str(saved['checks'].item()))
    if checks['nonanswer_block_comparisons'] != 48*done or checks['suppressed_update_calls'] != 12*done or checks.get('unhooked_clean_final_max') != 0:
        raise ValueError('Incomplete resume invariants')
    parity = checks['official_forward_parity']
    if parity['final_state_max_abs_error'] != 0 or not 0 <= parity['last_logit_max_abs_error'] <= 1e-4:
        raise ValueError('Missing official parity')
    return done, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=COHORTS, required=True)
    for name in ('manifest', 'audio-checksums', 'audio-root', 'source-root', 'model-path', 'whisper-path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=24)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error('Positive checkpoint interval required')
    rows = read_rows(args.manifest, None)
    ids = validate_cohort(args.dataset, args.manifest, rows)
    validate_audio_checksums(rows, args.audio_checksums, args.audio_root)
    import torch
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    logging.info('Loading audited LLaMA-Omni2 text path')
    torch, model, tokenizer, whisper, runtime = load_runtime(args.source_root, args.model_path, args.whisper_path, args.device)
    files = [Path(__file__), Path(inspect.getfile(load_runtime)), Path(inspect.getfile(install_hooks)), Path(inspect.getfile(validate_audio_checksums)), args.manifest, args.audio_checksums]
    files += sorted((args.source_root/'llama_omni2').rglob('*.py'))
    files += sorted(f for f in args.model_path.iterdir() if f.is_file() and f.suffix in ('.json', '.txt'))
    files.append(Path(inspect.getfile(type(model.model.layers[0]))))
    hashes = {str(f): sha256(f) for f in files}
    fingerprint = __import__('hashlib').sha256(json.dumps(dict(runtime=runtime, files=hashes, seed=1234), sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    resume_path = args.output.with_suffix('.resume.npz')
    block6, finals = [], []
    checks = {'nonanswer_block_comparisons': 0, 'suppressed_update_calls': 0}
    positions = None
    if args.resume and resume_path.exists():
        with np.load(resume_path, allow_pickle=False) as saved:
            done, checks = validate_replication_resume(saved, ids, fingerprint)
            block6, finals = list(saved['block6']), list(saved['final_answer'])
            positions = json.loads(str(saved['positions'].item()))
        logging.info('Resuming %d complete samples', done)
    elif resume_path.exists() or args.output.exists():
        raise FileExistsError('Output exists; use --resume or a fresh output path')
    blocks = model.model.layers
    if len(blocks) != 24:
        raise ValueError('Unexpected decoder depth')
    with torch.inference_mode():
        for i in range(len(block6), len(rows)):
            prefix, current = prepare_prefix(torch, model, tokenizer, whisper, args.audio_root/rows[i]['path'])
            if positions is not None and current != positions:
                raise ValueError('Input layout changed between samples')
            positions = current
            decision = positions['decision_position']
            unhooked = None
            if i == 0:
                checks['official_forward_parity'] = validate_forward_parity(torch, model, prefix)
                unhooked = model.model(inputs_embeds=prefix, use_cache=False, return_dict=True).last_hidden_state[0, decision].clone()
            clean_nonanswer, sample_b6, sample_final = {}, [], []
            for condition in CONDITIONS:
                captured = {}
                handles, calls = install_hooks(blocks, condition, decision, captured, clean_nonanswer, checks)
                try:
                    result = model.model(inputs_embeds=prefix, use_cache=False, return_dict=True)
                    final = result.last_hidden_state[0, decision]
                    sample_b6.append(captured['block6'])
                    sample_final.append(final.detach().float().cpu().numpy().copy())
                    if condition == 'clean' and unhooked is not None:
                        checks['unhooked_clean_final_max'] = float((final-unhooked).abs().max())
                        if not final.equal(unhooked):
                            raise AssertionError('Clean observer changed state')
                finally:
                    for handle in handles:
                        handle.remove()
                if calls != ([] if condition == 'clean' else list(TARGET_BLOCKS)):
                    raise AssertionError('Intervention layer calls differ')
                checks['suppressed_update_calls'] += len(calls)
            block6.append(np.stack(sample_b6)); finals.append(np.stack(sample_final))
            if (i+1) % args.checkpoint_every == 0 or i+1 == len(rows):
                save_npz(resume_path, dict(sample_ids=np.array(ids[:i+1]), conditions=np.array(CONDITIONS), block6=np.stack(block6), final_answer=np.stack(finals), fingerprint=np.array(fingerprint), checks=np.array(json.dumps(checks)), positions=np.array(json.dumps(positions))))
            logging.info('%s sample %d/%d complete', args.dataset, i+1, len(rows))
    arrays = dict(sample_ids=np.array(ids), conditions=np.array(CONDITIONS), block6=np.stack(block6), final_answer=np.stack(finals))
    for key in ('block6','final_answer'):
        if arrays[key].shape != (len(ids), 3, 896) or not np.isfinite(arrays[key]).all():
            raise AssertionError('Invalid endpoint vectors')
    if checks['nonanswer_block_comparisons'] != 48*len(ids) or checks['suppressed_update_calls'] != 12*len(ids):
        raise AssertionError('Incomplete intervention checks')
    save_npz(args.output, arrays)
    info = dict(model='LLaMA-Omni2-0.5B', dataset=args.dataset, n_samples=len(ids), conditions=CONDITIONS, target_blocks=TARGET_BLOCKS, seed=1234, positions=positions, checks=checks, runtime=runtime, sha256=hashes, output_sha256=sha256(args.output), resume_fingerprint=fingerprint, decoder_forwards=3*len(ids), additional_first_sample_decoder_validation_forwards=3, state_semantics={'block6':'Complete post-block6 answer residual','final_answer':'Complete answer after terminal RMSNorm'}, shape={k:list(v.shape) for k,v in arrays.items()}, scope='Fixed same-family model replication; no native LM-head behavior experiment.')
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
