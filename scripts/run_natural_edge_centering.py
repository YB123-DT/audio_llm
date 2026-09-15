#!/usr/bin/env python3
"""Leave-one-actor-out, label-free natural post-o_proj edge offset correction."""
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
CONDITIONS = {'clean': (), 'zero_noop': tuple(range(18,24)), 'center_joint': tuple(range(18,24)), 'center_19': (19,), 'center_23': (23,)}
SOURCE = Path('reports/2026-09-15-edge-representation/edge_vectors.npz')
METADATA = SOURCE.with_suffix('.csv')


def calibration(vectors, rows):
    """Uses actor and statement only: target actor never enters calibration."""
    actors = sorted({r['actor'] for r in rows})
    shifts, means, counts = {}, {}, {}
    for actor in actors:
        for statement in ('01', '02'):
            indices = [i for i,r in enumerate(rows) if r['actor'] != actor and r['statement'] == statement]
            if not indices:
                raise ValueError('Empty independent calibration set')
            means[actor,statement] = vectors[indices].astype(np.float64).mean(axis=0)
            counts[actor,statement] = len(indices)
        reference = .5*(means[actor,'01'] + means[actor,'02'])
        for statement in ('01','02'):
            shifts[actor,statement] = reference-means[actor,statement]
    return shifts, means, counts


def shifted_output(torch, output, shift, decision):
    result = output.clone()
    result[:,decision,:] += shift.to(device=output.device, dtype=output.dtype)
    return result


def run(args):
    data = np.load(SOURCE, allow_pickle=False)
    source_rows = list(csv.DictReader(METADATA.open()))
    lookup = {r['sample_id']:r for r in source_rows}
    source_rows = [lookup[str(s)] for s in data['sample_ids']]
    vectors, layer_ids = data['edge_updates'], data['layers'].tolist()
    if layer_ids != list(range(18,24)) or len(source_rows) != 192 or not np.isfinite(vectors).all():
        raise ValueError('Unexpected calibration artifact')
    shifts, means, counts = calibration(vectors, source_rows)
    torch, model, tokenizer, whisper = base.load_model(args)
    import transformers
    model.requires_grad_(False)
    rows = base.read_rows(args.manifest,args.limit)[args.start:]
    condition = base.FORCED_CHOICE_CONDITIONS['upper_prompt__lower_spaced']
    ids = [tokenizer.encode(condition[k],add_special_tokens=False) for k in ('happy_verbalizer','sad_verbalizer')]
    if ids != [[6247],[12421]]:
        raise ValueError('Single-token verbalizers changed')
    prefixes, positions = base._prepare_prefixes(torch,model,whisper,tokenizer,rows,args.ravdess_root,condition['prompt'])
    decision = positions['decision_position']
    layers, device = base._llm_layers(model), next(model.parameters()).device
    checks = dict(zero_noop_max=0., nondecision_rows_max=0., candidate_parity_max=0., historical_clean_parity_max=0., opposite_shift_max=max(float(np.max(np.abs(shifts[a,'01']+shifts[a,'02']))) for a,s in shifts))
    args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    norm_rows = []
    for (actor,statement),shift in shifts.items():
        for j,layer in enumerate(layer_ids):
            norm_rows.append(dict(actor=actor,statement=statement,layer=layer,n_calibration=counts[actor,statement],shift_norm=float(np.linalg.norm(shift[j])),mean_norm=float(np.linalg.norm(means[actor,statement][j]))))
    with (args.output_csv.parent/'calibration_norms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(norm_rows[0]),lineterminator='\n');w.writeheader();w.writerows(norm_rows)
    keys=sorted(shifts)
    np.savez_compressed(args.output_csv.parent/'calibration_means.npz', keys=np.asarray(keys), means=np.stack([means[k] for k in keys]),shifts=np.stack([shifts[k] for k in keys]),layers=layer_ids)
    fields=['sample_id','pair_id','actor','statement','repetition','intensity','emotion','condition','clean_margin','margin','shift_from_clean']
    with args.output_csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for index,row in enumerate(rows):
            prefix=prefixes[index].unsqueeze(0).to(device)
            mask=torch.ones(prefix.shape[:2],device=device,dtype=torch.bool)
            natural=None
            for name,selected in CONDITIONS.items():
                handles=[]
                def make_hook(layer):
                    shift=torch.as_tensor(shifts[row['actor'],row['statement']][layer_ids.index(layer)],device=device)
                    if name == 'zero_noop': shift=torch.zeros_like(shift)
                    def hook(_module,_inputs,output):
                        result=shifted_output(torch,output,shift,decision)
                        others=[i for i in range(output.shape[1]) if i!=decision]
                        checks['nondecision_rows_max']=max(checks['nondecision_rows_max'],float((result[:,others]-output[:,others]).abs().max()))
                        if not bool(torch.isfinite(result).all()): raise RuntimeError('Nonfinite update')
                        return result
                    return hook
                try:
                    for layer in selected: handles.append(layers[layer].self_attn.o_proj.register_forward_hook(make_hook(layer)))
                    with torch.inference_mode():
                        output=base._decoder_forward(model,prefix,mask)
                        logits=model.llm.lm_head(output.last_hidden_state[:,decision,:]).float()
                        margin=float(logits[0,6247]-logits[0,12421])
                        happy,sad=base._score_candidate_pairs(torch,model,prefix,mask,ids[0],ids[1],positions['prefix_length'])
                        checks['candidate_parity_max']=max(checks['candidate_parity_max'],abs(float(happy[0]-sad[0])-margin))
                finally:
                    for h in handles: h.remove()
                if not np.isfinite(margin): raise RuntimeError('Nonfinite score')
                if name=='clean':
                    natural=margin
                    checks['historical_clean_parity_max']=max(checks['historical_clean_parity_max'],abs(margin-float(lookup[row['sample_id']]['clean_margin'])))
                if name=='zero_noop': checks['zero_noop_max']=max(checks['zero_noop_max'],abs(margin-natural))
                writer.writerow({**{k:row[k] for k in fields[:7]},'condition':name,'clean_margin':natural,'margin':margin,'shift_from_clean':margin-natural})
            f.flush()
            if checks['zero_noop_max'] or checks['nondecision_rows_max'] or checks['historical_clean_parity_max']: raise RuntimeError(f'Exact validation failed {checks}')
            if max(checks.values())>args.self_patch_tolerance: raise RuntimeError(f'Tolerance failed {checks}')
            logging.info('completed sample %s/%s checks=%s',index+1,len(rows),checks)
    def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta=dict(n_samples=len(rows),conditions={k:list(v) for k,v in CONDITIONS.items()},positions=positions,checks=checks,seed=args.seed,model_frozen=all(not p.requires_grad for p in model.parameters()),donor_used=False,emotion_labels_used_for_calibration=False,calibration='For each target actor, both statement means from all OTHER 23 actors (92 samples per statement), fixed clean means; reference equal statement mean; no target actor samples; no label access in calibration()',intervention='o_proj forward output at decision row += reference_clean_mean - statement_clean_mean; others unchanged; later states endogenous; no bias subtraction needed for shifts',candidate_parity_samples=len(rows)*len(CONDITIONS),torch_version=torch.__version__,transformers_version=transformers.__version__,script_sha256=digest(__file__),input_sha256={str(p):digest(p) for p in (SOURCE,METADATA,args.manifest)},dependency_sha256={str(p):digest(p) for p in (edge.__file__,base.__file__)},checkpoint=str(args.checkpoint))
    args.output_csv.with_name(args.output_csv.stem+'_run.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run(base.parse_args())
