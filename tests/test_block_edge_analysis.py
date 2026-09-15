import pytest
from scripts.analyze_block_direct_edge import summarize
from scripts.run_block_direct_edge_restore import restore_schedules


def test_cumulative_increment_is_paired_difference():
    rows=[dict(actor=str(a),pair_id=str(a),emotion=e,condition=s,patch_effect=1,remaining_effect=1-len(layers),mediated_effect=len(layers)) for a in range(3) for e in ['happy','sad'] for s,layers in restore_schedules().items()]
    _,inc=summarize(rows)
    assert all(r['mean']==r['ci_low']==r['ci_high']==1 for r in inc)
    with pytest.raises(ValueError):summarize(rows[:-1])
