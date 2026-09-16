import json
import numpy as np
import pytest
from scripts.run_model_replication import CONDITIONS, validate_replication_resume


def test_replication_resume_binds_states_to_ids_and_verified_intervention():
    checks=dict(nonanswer_block_comparisons=48,suppressed_update_calls=12,unhooked_clean_final_max=0,official_forward_parity=dict(final_state_max_abs_error=0,last_logit_max_abs_error=0))
    state=dict(sample_ids=np.array(['a']),conditions=np.array(CONDITIONS),block6=np.zeros((1,3,896)),final_answer=np.zeros((1,3,896)),fingerprint=np.array('frozen'),checks=np.array(json.dumps(checks)))
    assert validate_replication_resume(state,['a','b'],'frozen')[0]==1
    with pytest.raises(ValueError,match='provenance'):
        validate_replication_resume(state,['a','b'],'changed')
    with pytest.raises(ValueError,match='identity'):
        validate_replication_resume(state,['b','a'],'frozen')
    for change in (dict(suppressed_update_calls=11),dict(official_forward_parity=dict(final_state_max_abs_error=.001,last_logit_max_abs_error=0))):
        altered=dict(state,checks=np.array(json.dumps(dict(checks,**change))))
        with pytest.raises(ValueError):validate_replication_resume(altered,['a','b'],'frozen')
    with pytest.raises(ValueError):
        validate_replication_resume(dict(state,block6=np.full((1,3,896),np.nan)),['a','b'],'frozen')
