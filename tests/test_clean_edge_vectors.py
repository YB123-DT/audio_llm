import subprocess
import sys
from pathlib import Path
import pytest
from scripts.extract_clean_edge_vectors import source_update
from scripts.run_audio_edge_restore import selected_attention


def test_source_updates_decompose_without_bias_or_renormalization():
    torch = pytest.importorskip('torch')
    gen = torch.Generator().manual_seed(812)
    q = torch.randn(1,4,7,3,generator=gen)
    k = torch.randn(1,2,7,3,generator=gen).repeat_interleave(2,dim=1)
    v = torch.randn(1,2,7,3,generator=gen).repeat_interleave(2,dim=1)
    weight = torch.randn(12,12,generator=gen)
    audio = [2,3,4]
    other = [0,1,5,6]
    a = source_update(torch,q,k,v,audio,6,None,3**-.5,weight)
    b = source_update(torch,q,k,v,other,6,None,3**-.5,weight)
    full = torch.nn.functional.linear(selected_attention(torch,q,k,v,[6],None,3**-.5),weight)
    torch.testing.assert_close(a+b,full,atol=1e-5,rtol=1e-5)
    weights = torch.softmax(q[:,:,6:7] @ k.transpose(-1,-2) * 3**-.5,dim=-1)
    explicit = (weights[:,:,:,audio] @ v[:,:,audio]).transpose(1,2).flatten(2)
    torch.testing.assert_close(a,torch.nn.functional.linear(explicit,weight),atol=1e-5,rtol=1e-5)
    assert torch.equal(source_update(torch,q,k,v,[],6,None,3**-.5,weight),torch.zeros_like(a))


def test_direct_entrypoint():
    script = Path(__file__).parents[1]/'scripts/extract_clean_edge_vectors.py'
    result = subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
