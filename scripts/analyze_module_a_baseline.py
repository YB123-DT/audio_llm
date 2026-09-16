#!/usr/bin/env python3
"""Unified fixed-protocol projector / complete answer state / native head baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
if __package__:
    from .analyze_edge_representation import folds, auc, clustered_metrics, write_csv
else:
    from analyze_edge_representation import folds, auc, clustered_metrics, write_csv


def align_legacy(rows, target_ids, vectors):
    ids = [r['sample_id'] for r in rows]
    indices = [int(r['representation_index']) for r in rows]
    if len(set(ids)) != len(ids) or len(set(target_ids)) != len(target_ids) or set(ids) != set(target_ids):
        raise ValueError('Legacy and current sample IDs differ or duplicate')
    if sorted(indices) != list(range(len(rows))) or len(vectors) != len(rows):
        raise ValueError('Legacy representation indices must form a complete permutation')
    lookup = dict(zip(ids,indices))
    return vectors[[lookup[s] for s in target_ids]]


def prepare(legacy_hidden, edge_vectors, model_config, destination):
    """Reconstruct the complete current answer state without another model forward."""
    import torch
    legacy = torch.load(legacy_hidden, map_location='cpu', weights_only=False)
    edge = np.load(edge_vectors, allow_pickle=False)
    config = json.loads(Path(model_config).read_text())
    run_path = Path(edge_vectors).with_name(Path(edge_vectors).stem+'_run.json')
    run = json.loads(run_path.read_text())
    if legacy['prompt'] != run['prompt'] or legacy['model_checkpoint'] != run['checkpoint']:
        raise ValueError('Legacy/current prompt or checkpoint path mismatch')
    eps = config['rms_norm_eps']
    h = torch.from_numpy(edge['final_decision_pre_norm'])
    weight = torch.from_numpy(edge['finalnorm_weight'])
    answer = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True)+eps) * weight).numpy()
    legacy_metadata = Path(legacy_hidden).with_name('metadata.csv')
    legacy_rows = list(csv.DictReader(legacy_metadata.open()))
    projector = align_legacy(legacy_rows, list(edge['sample_ids']), legacy['projector'].numpy())
    old = align_legacy(legacy_rows, list(edge['sample_ids']), legacy['llm_decision'][:, -1, :].numpy())
    provenance = dict(legacy_answer_rejected_margin_max=float(np.max(np.abs(old.astype(float) @ edge['d_lm'].astype(float)-edge['clean_margins']))), answer_source='Current clean extraction full final_decision_pre_norm reconstructed with original finalnorm_weight and official config RMSNorm epsilon; not edge update.', rms_norm_eps=eps, projector_source=str(legacy_hidden), projector_compatibility='Requires separate fresh projector provenance spotcheck; see interface audit.', prompt=legacy['prompt'], checkpoint=legacy['model_checkpoint'], sources_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (legacy_hidden, legacy_metadata, edge_vectors, model_config, run_path)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, sample_ids=edge['sample_ids'], projector=projector, answer_state=answer, d_lm=edge['d_lm'], clean_margins=edge['clean_margins'])
    destination.with_name('baseline_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def validate(data, rows, tolerance=1e-4):
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids) or ids != list(data['sample_ids']):
        raise ValueError('Metadata and vectors must have identical unique sample ID order')
    if all('representation_index' in r for r in rows) and [int(r['representation_index']) for r in rows] != list(range(len(rows))):
        raise ValueError('Legacy representation index order mismatch')
    if {r['emotion'] for r in rows} != {'happy', 'sad'}:
        raise ValueError('Expected happy/sad only')
    for key in ('projector', 'answer_state'):
        if data[key].shape != (len(rows), len(data['d_lm'])) or not np.isfinite(data[key]).all():
            raise ValueError('Invalid representation: '+key)
    parity = float(np.max(np.abs(data['answer_state'].astype(np.float64) @ data['d_lm'].astype(np.float64) - data['clean_margins'])))
    if not np.isfinite(parity) or parity > tolerance:
        raise ValueError(f'Complete answer state / native head parity failed: {parity}')
    return parity


def analyze(source, metadata, out, bootstrap=10000):
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    data = np.load(source, allow_pickle=False)
    rows = list(csv.DictReader(Path(metadata).open()))
    parity = validate(data, rows)
    y = np.array([int(r['emotion'] == 'happy') for r in rows])
    actors = np.array([r['actor'] for r in rows])
    unique = sorted(set(actors))
    draws = np.random.default_rng(20260915).integers(0, len(unique), (bootstrap, len(unique)))
    counts = np.array([np.bincount(d, minlength=len(unique)) for d in draws])
    summaries, predictions, fold_metrics, split_rows = [], [], [], []
    for regime in ('speaker', 'statement', 'joint'):
        split = list(folds(rows, regime))
        for fold, train, test in split:
            for subset, indices in [('train',train),('test',test)]:
                split_rows.extend(dict(regime=regime, fold=fold, subset=subset, sample_id=rows[i]['sample_id']) for i in indices)
        for name in ('projector', 'answer_state', 'native_lm'):
            scores = np.full(len(y), np.nan)
            coverage = np.zeros(len(y), dtype=int)
            fold_aucs = []
            for fold, train, test in split:
                if name == 'native_lm':
                    values = data['clean_margins'][test]
                    train_count = 0
                else:
                    x = data[name].astype(np.float64)
                    scaler = StandardScaler().fit(x[train])
                    clf = LogisticRegression(C=1., max_iter=5000, solver='lbfgs', random_state=20260915)
                    clf.fit(scaler.transform(x[train]), y[train])
                    if clf.n_iter_.max() >= 5000:
                        raise RuntimeError('Probe did not converge')
                    values = clf.decision_function(scaler.transform(x[test]))
                    train_count = len(train)
                scores[test] = values
                coverage[test] += 1
                fold_auc = auc(y[test], values)
                fold_aucs.append(fold_auc)
                fold_metrics.append(dict(object=name, regime=regime, fold=fold, n_train=train_count, n_test=len(test), accuracy=float(((values > 0)==y[test]).mean()), roc_auc=fold_auc))
                predictions.extend(dict(object=name, regime=regime, fold=fold, sample_id=rows[i]['sample_id'], actor=rows[i]['actor'], statement=rows[i]['statement'], emotion=rows[i]['emotion'], score=float(value), prediction='happy' if value > 0 else 'sad') for i,value in zip(test,values))
            if not np.all(coverage == 1) or not np.isfinite(scores).all():
                raise ValueError('OOF coverage failed')
            summaries.extend(dict(object=name,regime=regime,**m) for m in clustered_metrics(y,scores,actors,counts))
            summaries.append(dict(object=name,regime=regime,metric='mean_within_fold_auc',mean=float(np.mean(fold_aucs)),ci_low='',ci_high=''))
    out.mkdir(parents=True, exist_ok=True)
    for filename,values in [('summary',summaries),('oof_predictions',predictions),('fold_metrics',fold_metrics),('fold_membership',split_rows)]:
        write_csv(out / (filename+'.csv'), values)
    analysis = dict(n_samples=len(rows), n_actors=len(unique), answer_head_parity_max=parity, input_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(), metadata_sha256=hashlib.sha256(Path(metadata).read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sklearn_version=sklearn.__version__, bootstrap=bootstrap, seed=20260915, protocol='Train-only StandardScaler + LogisticRegression C=1 lbfgs max_iter=5000; happy=1; no tuning; zero margin predicts sad.', objects='Projector audio-token mean; complete final RMSNorm answer-position state; native single-token happy-minus-sad log likelihood margin.', uncertainty='Actor-cluster percentile 95%, fixed fitted folds, no bootstrap refitting, unadjusted; within-fold AUC descriptive.', native='No external fit: identical native scores reused across regimes, with only within-fold grouping changed.', folds='Speaker LOAO; statement bidirectional; joint train excludes both test actor and test statement. Each sample exactly once per object/regime.', pooled_auc='External OOF scores arise from different fitted models; pooled AUC may reflect fold offsets, so mean within-fold AUC is also reported.')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    lookup={(r['object'],r['regime'],r['metric']):r for r in summaries}
    table=['| Readout | Speaker Acc / pooled AUC / fold AUC | Statement Acc / pooled AUC / fold AUC | Joint Acc / pooled AUC / fold AUC |','|---|---:|---:|---:|']
    for name in ('projector','answer_state','native_lm'):
        cells=[]
        for regime in ('speaker','statement','joint'):
            cells.append(' / '.join(f"{lookup[name,regime,m]['mean']:.4f}" for m in ('accuracy','roc_auc','mean_within_fold_auc')))
        table.append('| '+name+' | '+' | '.join(cells)+' |')
    (out/'baseline-table.md').write_text('\n'.join(table)+'\n')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representations',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--legacy-hidden',type=Path)
    parser.add_argument('--edge-vectors',type=Path)
    parser.add_argument('--model-config',type=Path)
    args=parser.parse_args()
    if args.legacy_hidden:
        if not args.edge_vectors or not args.model_config:
            parser.error('--legacy-hidden requires --edge-vectors and --model-config')
        prepare(args.legacy_hidden,args.edge_vectors,args.model_config,args.representations)
    analyze(args.representations,args.metadata,args.output_dir,args.bootstrap)
