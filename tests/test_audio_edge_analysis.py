import pytest
from scripts.analyze_audio_edge_restore import summarize,verify,QUERIES


def test_query_contrasts_preserve_interaction():
    values=dict(decision=2,audio=.5,end_marker=.1,non_decision=.6,all=2.3)
    rows=[dict(actor=str(a),pair_id=str(a),emotion=e,condition=f'edge_{q}_joint',mediated_effect=values[q],patch_effect=3,remaining_effect=3-values[q]) for a in range(3) for e in ['happy','sad'] for q in QUERIES]
    summary,contrast=summarize(rows)
    c={r['contrast']:r for r in contrast}
    assert c['all_minus_decision_minus_non_decision']['mean']==pytest.approx(-.3)
    assert c['decision_minus_all']['ci_low']==pytest.approx(-.3)
    with pytest.raises(ValueError):summarize(rows[:-1])


def test_all_query_previous_source_reference_parity():
    rows=[dict(sample_id='s',condition=f'edge_{q}_joint',clean_margin=0,patched_margin=1,restored_margin=.5) for q in QUERIES]
    ref=[dict(sample_id='s',condition='v_audio_joint',clean_margin=0,patched_margin=1,restored_margin=.5)]
    assert verify(rows,ref)['checks']['previous_audio_source_restored_margin_max_abs_error']==0
    ref[0]['restored_margin']=.6
    with pytest.raises(ValueError):verify(rows,ref)
