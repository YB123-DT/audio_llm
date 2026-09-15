from scripts.analyze_source_value_restore import summarize, SOURCES
import pytest


def test_matched_contrast_and_nonadditivity():
    effects=dict(audio=2,prompt=0,other=1,non_audio=1,decision=.4,post_audio_marker=.6,all=2.5)
    rows=[dict(actor=str(a),pair_id=str(a),emotion=e,condition=f'v_{s}_joint',patch_effect=3,remaining_effect=3-effects[s],mediated_effect=effects[s]) for a in range(3) for e in ['happy','sad'] for s in SOURCES]
    summary,contrasts=summarize(rows)
    by={r['contrast']:r for r in contrasts}
    assert by['audio_minus_all']['mean']==-.5
    assert by['all_minus_partition_sum']['ci_low']==-.5
    assert next(r for r in summary if r['source']=='prompt' and r['metric']=='mediated_effect')['ci_high']==0
    with pytest.raises(ValueError):
        summarize(rows[:-1])


def test_prior_all_v_and_structural_null_checks():
    from scripts.analyze_source_value_restore import verify_sources
    rows=[dict(sample_id='s',condition=f'v_{s}_joint',clean_margin=0,patched_margin=1,restored_margin=(1 if s=='prompt' else .5)) for s in SOURCES]
    ref=[dict(sample_id='s',condition='v_joint',clean_margin=0,patched_margin=1,restored_margin=.5)]
    assert verify_sources(rows,ref)['numerical_checks']['prior_all_v_max_abs_error']==0
    ref[0]['restored_margin']=.7
    with pytest.raises(ValueError,match='verification failed'):
        verify_sources(rows,ref)
