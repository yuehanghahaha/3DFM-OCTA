"""Measure an equivalent Conv3d memory layout before fixing the training protocol."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run256 import make,update,read
from engine import write_json
import numpy as np
import torch

torch.set_num_threads(2)
config=read(ROOT/'suite.json')
selection=read(ROOT/'data/selection.json')
sample=next(s for s in selection['patches'] if s['split']=='train')
with np.load(ROOT/sample['file'],allow_pickle=False) as f:
    source,target=[torch.from_numpy(f[k])[None,None].to('cuda',memory_format=torch.channels_last_3d) for k in ('oct','octa')]
name='03_fm3d_zproj'
model=make(name,torch.device('cuda'),config).to(memory_format=torch.channels_last_3d)
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.)
scaler=torch.amp.GradScaler('cuda',init_scale=1024)
torch.cuda.reset_peak_memory_stats()
tick=time.perf_counter()
rows=[]
try:
    for _ in range(2):
        loss,success=update(model,name,source,target,torch.tensor([.37],device='cuda'),optimizer,scaler)
        rows.append(dict(losses=loss,applied=success))
    torch.cuda.synchronize()
    result=dict(status='passed',seconds_per_attempt=(time.perf_counter()-tick)/2,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),updates=rows)
except torch.cuda.OutOfMemoryError as error:result=dict(status='out_of_memory',error=str(error))
write_json(ROOT/'verification/layout_03.json',result)
print(json.dumps(result),flush=True)
