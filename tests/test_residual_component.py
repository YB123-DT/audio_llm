"""Pairing and clustered contrast checks independent of model runtime."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_residual_component_restore import grouped_pairs, estimate


def test_pair_mean_uses_both_directions_and_rejects_duplicate():
    rows = [dict(actor='01', pair_id='p', emotion=e, effect=v) for e,v in [('happy', 2),('sad', 4)]]
    assert grouped_pairs(rows, 'effect') == {('01','p'): 3}
    with pytest.raises(ValueError):
        grouped_pairs(rows + rows[:1], 'effect')


def test_paired_cluster_contrast_cancels_shared_actor_variation():
    a = {(str(i), str(j)): 100*i+j for i in range(4) for j in range(4)}
    m = {k: v-2 for k,v in a.items()}
    draws = np.random.default_rng(42).integers(0,4,(1000,4))
    result = estimate({k:a[k]-m[k] for k in a}, draws, list(map(str,range(4))))
    assert result['mean'] == result['ci_low'] == result['ci_high'] == 2
    assert result['n_pairs'] == 16


def test_end_to_end_gate_uses_paired_attention_minus_mlp(tmp_path):
    import csv
    import json
    from analyze_residual_component_restore import analyze
    rows, attrs = [], []
    for actor in ['01','02']:
        for emotion in ['happy','sad']:
            meta = dict(actor=actor, pair_id=actor, emotion=emotion)
            for component in ['attention','mlp','qk','v']:
                for suffix in ['18','19','20','21','22','23','joint']:
                    # Both components individually positive; MLP is larger.
                    value = {'attention':1, 'mlp':2, 'qk':.5, 'v':2.5}[component]
                    rows.append(dict(meta, condition=f'{component}_{suffix}', patch_effect=3,
                                     remaining_effect=3-value, mediated_effect=value))
            for layer in range(24):
                attrs.append(dict(meta, layer_index=layer, delta_raw=0, delta_lens=0, clean_lens=0))
    for name, data in [('effects.csv',rows),('attrs.csv',attrs)]:
        with (tmp_path/name).open('w') as f:
            w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    analyze(tmp_path/'effects.csv',tmp_path/'attrs.csv',tmp_path/'analysis')
    gate=json.loads((tmp_path/'analysis/attention_gate.json').read_text())
    assert gate['joint_attention_mediation']['ci_low'] == 1
    assert gate['joint_attention_minus_mlp']['ci_high'] == -1
    assert gate['attention_wins'] is False
    with (tmp_path/'analysis/component_contrasts.csv').open() as f:
        contrast = next(r for r in csv.DictReader(f) if r['condition']=='joint' and r['metric']=='qk_minus_v_mediation')
    assert float(contrast['ci_high']) == -2
    assert (tmp_path/'analysis/qkv_restore.png').is_file()
