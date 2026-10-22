"""Read-only checkpoint diagnosis on predetermined train/validation patches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_pair
from .model import VelocityUNet3D


def stats(a, b):
    a, b = a.detach().float(), b.detach().float()
    e = a-b
    mse = e.square().mean().item()
    out = dict(mae=e.abs().mean().item(), mse=mse, psnr=-10*np.log10(max(mse,1e-12)),
               pred_mean=a.mean().item(), target_mean=b.mean().item(), pred_std=a.std().item(), target_std=b.std().item())
    for name, axis in [('mean',None),('max',0)]:
        x,y = a[0,0],b[0,0]
        x,y = (x.mean(0),y.mean(0)) if axis is None else (x.max(0).values,y.max(0).values)
        out[name+'_projection_mae'] = (x-y).abs().mean().item()
        out[name+'_projection_corr'] = float(np.corrcoef(x.cpu().numpy().ravel(),y.cpu().numpy().ravel())[0,1]) if x.std()>0 and y.std()>0 else None
        out[name+'_projection_std_ratio'] = (x.std()/y.std().clamp_min(1e-12)).item()
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--saved-validation',type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config']
    model=VelocityUNet3D(cfg['base_channels'],cfg['condition']).cuda().eval()
    model.load_state_dict(ck['model'])
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ['train','validation']:
        rows.extend(sorted([r for r in ck['manifest'] if r['split']==split],key=lambda r:r['case_id'])[:2])
    result=dict(checkpoint_epoch=ck['epoch'], config=cfg,
        protocol='First2 sorted train and first2 sorted validation IDs; fixed full-depth64x64 windows at(y,x)=(120,120),(64,64); FP32; no test data; GT>0.1 mask is intensity diagnostic only, not vessel segmentation.',cases=[])
    with torch.inference_mode():
        for row in rows:
            source,target=load_pair(row['path'],cfg['axis_order'],cfg['intensity_scale'],cfg['depth'])
            for loc in [120,64]:
                x=source[None,:,:,loc:loc+64,loc:loc+64].cuda()
                y=target[None,:,:,loc:loc+64,loc:loc+64].cuda()
                case=dict(case_id=row['case_id'],split=row['split'],yx=loc,oracle=[],rollout=[])
                mask=y>0.1
                case['target_above_0p1_fraction']=mask.float().mean().item()
                case['zero_baseline']=stats(torch.zeros_like(y),y)
                case['source_baseline']=stats(x,y)
                for t in [0,.001,.01,.025,.05,.1,.25,.5,.75,.9,1]:
                    state=(1-t)*x+t*y
                    v=model(state,torch.tensor([t],device=x.device),x)
                    err=(v-(y-x)).square()
                    endpoint=state+(1-t)*v
                    case['oracle'].append(dict(t=t,velocity_mse=err.mean().item(),
                        signal_mse=err[mask].mean().item(), low_signal_mse=err[~mask].mean().item(),
                        signal_error_fraction=(err[mask].sum()/err.sum()).item(), endpoint=stats(endpoint.clamp(0,1),y)))
                # Counterfactual: keep OCT fixed but insert a spatially flipped OCTA into x_t.
                alt=y.flip(-1)
                t=.5
                mixed=(1-t)*x+t*alt
                v=model(mixed,torch.tensor([t],device=x.device),x)
                end=(mixed+(1-t)*v).clamp(0,1)
                case['counterfactual']=dict(against_injected=stats(end,alt),against_paired=stats(end,y),
                    warning='Off-path diagnostic, not a valid generation result or proof of exact learned algebra')
                arrays=dict(source=x.cpu().numpy(),target=y.cpu().numpy())
                for steps in [1,8,32,128]:
                    state=x.clone()
                    for n in range(steps):
                        v=model(state,torch.tensor([n/steps],device=x.device),x)
                        state=state+v/steps
                    score=stats(state.clamp(0,1),y)
                    score.update(steps=steps,negative_fraction=(state<0).float().mean().item(),above_one_fraction=(state>1).float().mean().item())
                    case['rollout'].append(score)
                    arrays['prediction_'+str(steps)]=state.clamp(0,1).cpu().numpy()
                    if steps==8:
                        with torch.autocast('cuda'):
                            state_amp=x.clone()
                            for n in range(8):
                                state_amp=state_amp+model(state_amp,torch.tensor([n/8],device=x.device),x).float()/8
                        case['amp_vs_fp32']=stats(state_amp.clamp(0,1),state.clamp(0,1))
                if row['split']=='validation' and args.saved_validation:
                    path=args.saved_validation/(row['case_id']+'.npz')
                    with np.load(path) as f:
                        np.testing.assert_allclose(f['octa_gt'],target[0].numpy(),rtol=0,atol=0)
                        saved=torch.from_numpy(f['octa'][None,None,:,loc:loc+64,loc:loc+64]).cuda()
                    case['saved_whole_volume_patch']=stats(saved,y)
                    case['saved_vs_direct8']=stats(saved,torch.from_numpy(arrays['prediction_8']).cuda())
                    arrays['saved_whole_volume_patch']=saved.cpu().numpy()
                np.savez_compressed(args.output/(row['case_id']+'_'+str(loc)+'.npz'),**arrays)
                result['cases'].append(case)
                (args.output/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(row['case_id'],row['split'],loc,'oracle_mse_t0_t05',case['oracle'][0]['velocity_mse'],case['oracle'][7]['velocity_mse'],
                      'rollout_mae',[round(r['mae'],6) for r in case['rollout']],flush=True)


if __name__=='__main__':
    main()
