import numpy as np
import pytest
from scripts.run_natural_edge_centering import calibration, shifted_output


def test_calibration_excludes_actor_and_does_not_read_labels():
    rows=[dict(actor=a,statement=s) for a in ('01','02','03') for s in ('01','02')]
    x=np.arange(12.).reshape(6,1,2)
    shifts,means,counts=calibration(x,rows)
    np.testing.assert_allclose(means['01','01'],x[[2,4]].mean(axis=0))
    changed=x.copy();changed[:2]+=10000
    newer=calibration(changed,rows)[0]
    np.testing.assert_array_equal(newer['01','01'],shifts['01','01'])
    np.testing.assert_allclose(shifts['01','01']+shifts['01','02'],0)
    assert counts['01','01']==2
    labelled=[dict(r,emotion='arbitrary') for r in rows]
    for key,value in calibration(x,labelled)[0].items():np.testing.assert_array_equal(value,shifts[key])


def test_output_shift_isolation_and_noop():
    torch=pytest.importorskip('torch')
    x=torch.randn(2,5,7)
    shift=torch.randn(7)
    changed=shifted_output(torch,x,shift,3)
    assert torch.equal(changed[:,[0,1,2,4]],x[:,[0,1,2,4]])
    torch.testing.assert_close(changed[:,3],x[:,3]+shift)
    assert torch.equal(shifted_output(torch,x,torch.zeros(7),3),x)
