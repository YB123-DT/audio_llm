import pytest
from scripts.analyze_full_audio_sweep import summarize


def fixture_rows():
    return [dict(actor=str(a),pair_id=f'{a}-{p}',layer=l,emotion=e,patch_effect=(l+1)*(a+1)/10) for a in range(4) for p in range(2) for l in range(24) for e in ['happy','sad']]


def test_cluster_means_and_simultaneous_band():
    summary,contrast,meta=summarize(fixture_rows(),1000)
    assert summary[0]['mean']==pytest.approx(.25)
    assert summary[23]['mean']==pytest.approx(6.)
    assert contrast[17]['mean']==0
    assert meta['n_pairs']==8
    assert meta['simultaneous_radius']>0
    assert all(r['simultaneous_low']<=r['mean']<=r['simultaneous_high'] for r in summary)


def test_reject_duplicate_or_missing():
    rows=fixture_rows()
    with pytest.raises(ValueError,match='Duplicate'):summarize(rows+[rows[0]])
    with pytest.raises(ValueError,match='Incomplete'):summarize(rows[1:])
