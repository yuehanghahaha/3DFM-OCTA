"""Small controlled memorization test, NOT a validation or proposed production model."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import VelocityUNet3D,flow_loss
from .data import load_pair
from .diagnose_voxel import stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config']
    row=sorted([r for r in ck['manifest'] if r['split']=='train'],key=lambda r:r['case_id'])[0]
    x,y=load_pair(row['path'],cfg['axis_order'],255,256)
    x,y=[v[None,:,:,64:128,64:128].cuda() for v in [x,y]]
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(case_id=row['case_id'],split='train',patch_yx=64,updates_per_arm=120,
        protocol='Both arms start from epoch90 with fresh AdamW lr1e-4, no weight decay, clip1; same fixed training patch, FP32. Original uniform-t versus alternating exactt0/uniform-t. No test cases; measures memorization only, not generalization.',arms=[])
    for arm in ['uniform','half_t0']:
        torch.manual_seed(2026)
        m=VelocityUNet3D(cfg['base_channels'],cfg['condition']).cuda()
        m.load_state_dict(ck['model'])
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0)
        history=[]
        for step in range(121):
            if step in [0,40,120]:
                m.eval()
                with torch.inference_mode():
                    direct=(x+m(x,torch.zeros(1,device=x.device),x)).clamp(0,1)
                    state=x.clone()
                    for j in range(8):state=state+m(state,torch.tensor([j/8],device=x.device),x)/8
                    hist=dict(updates=step,direct_t0=stats(direct,y),rollout8=stats(state.clamp(0,1),y))
                    history.append(hist)
                    np.savez_compressed(a.output/(arm+'_'+str(step)+'.npz'),target=y.cpu().numpy(),direct=direct.cpu().numpy(),rollout=state.clamp(0,1).cpu().numpy())
                    print(arm,step,'direct corr',hist['direct_t0']['max_projection_corr'],'rollout corr',hist['rollout8']['max_projection_corr'],flush=True)
                m.train()
            if step==120:break
            opt.zero_grad(set_to_none=True)
            t=torch.rand(1,device=x.device)
            if arm=='half_t0' and step%2==0:t.zero_()
            loss=flow_loss(m,x,y,time=t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
            opt.step()
        result['arms'].append(dict(name=arm,history=history))
        (a.output/'fit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        del m,opt
        torch.cuda.empty_cache()


if __name__=='__main__':main()
