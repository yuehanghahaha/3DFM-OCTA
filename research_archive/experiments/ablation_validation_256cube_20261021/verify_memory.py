"""Check recomputation against the original equations before large-cube training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import torch
from memory_model import MemoryUNet,verify_equivalence
from fm3d.model import VelocityUNet3D
from engine import objectives,write_json

torch.set_num_threads(2)
results=[verify_equivalence()]
for representation in ('voxel','haar3d'):
    torch.manual_seed(73)
    channels=1 if representation=='voxel' else 8
    original=VelocityUNet3D(4,True,channels)
    memory=MemoryUNet(4,True,channels)
    memory.load_state_dict(original.state_dict())
    source,target=torch.rand(1,1,16,16,16),torch.rand(1,1,16,16,16)
    arm=dict(representation=representation,lambda_projection=.1,lambda_reconstruction=1.)
    a,_=objectives(original,source,target,torch.tensor([.37]),arm,8,False)
    b,_=objectives(memory,source,target,torch.tensor([.37]),arm,8,True)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    a.backward();b.backward()
    for (name,p),(_,q) in zip(original.named_parameters(),memory.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=4e-6,rtol=5e-4,msg=name)
    results.append(dict(representation=representation,free8_full_loss_and_gradients_match=True))
write_json(ROOT/'verification/memory_equivalence.json',dict(status='passed',checks=results))
print('PASS: same outputs, free8 objectives and all parameter gradients')
