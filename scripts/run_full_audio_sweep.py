#!/usr/bin/env python3
"""Frozen opposite-emotion post-block audio replacement over all24 decoder blocks."""
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
if __package__:
    from .run_audio_edge_restore import base
else:
    from run_audio_edge_restore import base


def run(args):
    torch, model, tokenizer, whisper = base.load_model(args)
    import transformers
    model.requires_grad_(False)
    rows = base.read_rows(args.manifest, args.limit)[args.start:]
    pairs = base._pair_indices(rows)
    if len(pairs)*2 != len(rows):
        raise ValueError('Complete matched pairs required')
    layers = base._llm_layers(model)
    if len(layers) != 24:
        raise ValueError('Expected24 blocks')
    condition = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    ids = [tokenizer.encode(condition[k], add_special_tokens=False) for k in ('happy_verbalizer','sad_verbalizer')]
    if ids != [[6247],[12421]]:
        raise ValueError(f'Unexpected labels {ids}')
    prefixes, positions = base._prepare_prefixes(torch, model, whisper, tokenizer, rows, args.ravdess_root, condition['prompt'])
    device = next(model.parameters()).device
    checks = dict(self_patch_max=0., final_layer_effect_max=0., clean_candidate_parity_max=0., patched_candidate_parity_max=0.)
    fields = ['sample_id','pair_id','actor','statement','repetition','intensity','emotion','donor_sample_id','layer','clean_margin','patched_margin','donor_sign','patch_effect']
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for pi,pair in enumerate(pairs):
            indices = [pair['happy_index'],pair['sad_index']]
            donor = base._capture_all_donors(torch,model,[prefixes[i] for i in indices],positions,tuple(range(24)),device,1)
            for within,index in enumerate(indices):
                row = rows[index]
                prefix = prefixes[index].unsqueeze(0).to(device)
                mask = torch.ones(prefix.shape[:2],device=device,dtype=torch.bool)
                def forward(layer=None,state=None):
                    handle = None
                    try:
                        if layer is not None:
                            handle = layers[layer].register_forward_hook(base._patch_hook(state,'audio_tokens',positions['audio_start'],positions['audio_length'],positions['decision_position']))
                        with torch.inference_mode():
                            result = base._decoder_forward(model,prefix,mask)
                            logits = model.llm.lm_head(result.last_hidden_state[:,positions['decision_position'],:]).float()
                            value = float(logits[0,6247]-logits[0,12421])
                            if not math.isfinite(value):raise RuntimeError('Nonfinite margin')
                            return value
                    finally:
                        if handle is not None:handle.remove()
                clean = forward()
                h,s = base._score_candidate_pairs(torch,model,prefix,mask,ids[0],ids[1],positions['prefix_length'])
                checks['clean_candidate_parity_max'] = max(checks['clean_candidate_parity_max'],abs(float(h[0]-s[0])-clean))
                for layer in range(24):
                    state = donor[layer]['audio_tokens'][1-within:2-within]
                    patched = forward(layer,state)
                    if pi == 0:
                        checks['self_patch_max'] = max(checks['self_patch_max'],abs(forward(layer,donor[layer]['audio_tokens'][within:within+1])-clean))
                        h,s = base._score_candidate_pairs(torch,model,prefix,mask,ids[0],ids[1],positions['prefix_length'],layer_module=layers[layer],donor=state,patch_kind='audio_tokens',**{k:positions[k] for k in ('audio_start','audio_length','decision_position')})
                        checks['patched_candidate_parity_max'] = max(checks['patched_candidate_parity_max'],abs(float(h[0]-s[0])-patched))
                    if layer == 23:checks['final_layer_effect_max'] = max(checks['final_layer_effect_max'],abs(patched-clean))
                    sign = 1 if row['emotion']=='sad' else -1
                    result = {k:row[k] for k in fields[:7]}
                    writer.writerow(dict(result,donor_sample_id=rows[indices[1-within]]['sample_id'],layer=layer,clean_margin=clean,patched_margin=patched,donor_sign=sign,patch_effect=sign*(patched-clean)))
                f.flush()
            if any(v>args.self_patch_tolerance or not math.isfinite(v) for v in checks.values()):raise RuntimeError(checks)
            logging.info('completed pair %d/%d checks=%s',pi+1,len(pairs),checks)
    metadata = dict(n_samples=len(rows),n_pairs=len(pairs),layers=list(range(24)),patch_boundary='post-block output residual; audio positions only',positions=positions,prompt=condition['prompt'],happy_token_ids=ids[0],sad_token_ids=ids[1],numerical_checks=checks,self_patch_samples=2,patched_candidate_parity_samples=2,clean_candidate_parity_samples=len(rows),model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),torch_version=torch.__version__,transformers_version=transformers.__version__,checkpoint=str(args.checkpoint),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dependency_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('run_audio_edge_restore.py','run_source_value_restore.py','run_activation_patching.py')})
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run(base.parse_args())
