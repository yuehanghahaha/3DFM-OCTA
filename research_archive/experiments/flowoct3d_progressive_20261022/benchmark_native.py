from pathlib import Path
from types import SimpleNamespace
import sys
import time
import json
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE/'flowoct_reproduction'))
sys.path.insert(0, str(WORKSPACE))
import torch
import numpy as np
from flowoct.model import FlowOCT
from flowoct3d.utils import write_json
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(20261021)
print('Creating native FlowOCT', flush=True)
opt = SimpleNamespace(gpu_ids=[0], isTrain=True, checkpoints_dir=str(ROOT), name='benchmark',
                      lr=2e-4, beta=.5, train_slices=0, network_chunk=0)
model = FlowOCT(opt)
p = WORKSPACE/'experiments/flowoct3d_full_20261022/data/patches/10301_y064_x064.npz'
with np.load(p) as d:
    a, b = [torch.from_numpy(d[k])[None, None].cuda().permute(0, 1, 3, 4, 2).contiguous() for k in ('oct', 'octa')]
model.set_input({'A': a, 'B': b})
torch.cuda.reset_peak_memory_stats()
results = []
for i in range(4):
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.forward()
    torch.cuda.synchronize()
    results.append(time.perf_counter()-start)
    print('Native update', i+1, results[-1], flush=True)
r = dict(seconds_per_update=results, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         losses=dict(model.get_current_loss()), parameters=sum(p.numel() for p in model.netG.parameters()),
         gpu=torch.cuda.get_device_name())
write_json(ROOT/'verification/native_benchmark.json', r)
print(json.dumps(r), flush=True)
