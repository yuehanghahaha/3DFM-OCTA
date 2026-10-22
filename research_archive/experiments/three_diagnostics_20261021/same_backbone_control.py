"""Capacity-matched supervised control, with no FM states or integration."""
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_diagnostics import ROOT, SEED, load_samples, maps, module, scores, seed, stamp, write_json


class DirectRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = module('voxel', 'model').VelocityUNet3D(16, True)

    def forward(self, oct_volume):
        time_zero = torch.zeros(len(oct_volume), device=oct_volume.device)
        return oct_volume.float() + self.network(oct_volume, time_zero, oct_volume).float()


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    output = ROOT / 'experiment1_same_backbone'
    output.mkdir(exist_ok=False)
    write_json(output / 'protocol.json', dict(created=stamp(),
        reason='Additional control avoids confounding the no-FM comparison with the reduced parameter count and changed architecture of the plain U-Net. The original compact control and all its results are retained.',
        formula='prediction=OCT+network(OCT,t=0,OCT); loss=MSE(prediction,OCTA). No FM path, random t, target-containing state, or ODE is used.',
        model='Exact frozen production voxel U-Net; random initialization; not a fine-tune',
        seed=SEED, successful_updates=2400, optimizer='AdamW, lr2e-4, weight_decay0, clip1, AMP FP16 with dynamic scaler init1024',
        update_schedule='Constant lr; no validation-based selection',
        scope='Single training patch only. This control tests the existing backbone at t=0; it is not a trained multi-step FM model.'))
    write_json(ROOT / 'status.json', dict(state='running', stage='same_backbone_supervised_control', updated=stamp()))
    samples = load_samples()
    seed()
    model = DirectRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    x,y=(samples[0][key].to(device) for key in ('x','y'))
    constant=scores(torch.full_like(y, y.mean()), y)
    history=[]
    completed=skipped=0
    tick=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def evaluate_step():
        model.eval()
        with torch.inference_mode():
            prediction=model(x)
        record=dict(update=completed, seconds=time.perf_counter()-tick,
                    raw=scores(prediction,y), final_clamp=scores(prediction.clamp(0,1),y))
        history.append(record)
        write_json(output/'history.json',history)
        maps(output/f'maps_{completed:04d}.npz',prediction.clamp(0,1))
        print('SAME_BACKBONE',completed,json.dumps(record['raw']), 'seconds', round(record['seconds'],1),flush=True)

    evaluate_step()
    while completed<2400:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',enabled=True):
            prediction=model(x)
            loss=F.mse_loss(prediction.float(),y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite supervised loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
        previous_scale=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale()<previous_scale:
            skipped+=1
            if skipped>100:
                raise FloatingPointError('Repeated AMP overflow')
            continue
        completed+=1
        if completed%200==0:
            if not math.isfinite(float(grad_norm)) or not float(grad_norm)>0:
                raise FloatingPointError('No finite nonzero gradients')
            evaluate_step()
        if completed%800==0:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                            completed=completed,seed=SEED),output/f'update_{completed:04d}.pt')
    model.eval()
    evaluation=[]
    with torch.inference_mode():
        for sample in samples:
            prediction=model(sample['x'].to(device)).cpu()
            evaluation.append(dict(**sample['meta'],raw=scores(prediction,sample['y']),
                                   final_clamp=scores(prediction.clamp(0,1),sample['y']),
                                   trained_on_this_patch=sample is samples[0]))
            maps(output/(sample['meta']['id']+'.npz'),prediction.clamp(0,1))
    final=history[-1]['raw']
    passed=final['max_projection_pcc']>=.9 and final['max_projection_ssim']>=.8 and final['mse']<=.1*constant['mse']
    write_json(output/'result.json',dict(complete=True,updates=completed,successful_updates=completed,
        skipped_amp_updates=skipped,parameter_count=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        memorization_threshold_passed=passed,constant_mean_predictor=constant,evaluation=evaluation))
    write_json(ROOT/'status.json',dict(state='complete',stages=3,additional_capacity_control=True,completed=stamp()))
    print('SAME BACKBONE CONTROL COMPLETE',stamp(),flush=True)


if __name__=='__main__':
    main()
