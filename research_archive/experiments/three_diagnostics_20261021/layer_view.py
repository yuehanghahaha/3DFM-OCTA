"""Post-hoc ILM-OPL checks on preselected train/validation patches."""
import json

import numpy as np
import torch
from skimage.metrics import structural_similarity

from run_diagnostics import (ROOT, SERVER, PlainUNet3D, decode, encode, load_model,
                             pcc, rollout, sha256, stamp, write_json)
from same_backbone_control import DirectRegressor


def project(a, layers):
    centers=((np.arange(a.shape[0])+.5)*640/a.shape[0]-.5)[:,None,None]
    mask=(centers>=layers[0]-1)&(centers<=layers[2]-1)
    return np.where(mask,a,0).max(0), int((mask.sum(0)==0).sum())


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda:0')
    output=ROOT/'posthoc_layers'
    output.mkdir(exist_ok=False)
    prior=SERVER/'fm3d_validation_20261021'
    selection=json.loads((prior/'data/selection.json').read_text())
    small=PlainUNet3D().to(device).eval()
    small.load_state_dict(torch.load(ROOT/'experiment1_supervised/update_2400.pt',map_location='cpu',weights_only=False)['model'])
    matched=DirectRegressor().to(device).eval()
    matched.load_state_dict(torch.load(ROOT/'experiment1_same_backbone/update_2400.pt',map_location='cpu',weights_only=False)['model'])
    voxel,haar=load_model('voxel',device),load_model('haar',device)
    results=[]
    for split,cid in [('train','10301'),('validation','10302')]:
        for loc in (64,120):
            identity=f'{split}_{cid}_y{loc:03d}_x{loc:03d}'
            record=next(r for r in selection['patches'] if r['case_id']==cid and r['origin_ZYX']==[0,loc,loc])
            cached=prior/record['file']
            assert sha256(cached)==record['sha256']
            with np.load(ROOT/'inputs'/(identity+'.npz')) as f:
                x,y=f['oct'],f['octa']
            with np.load(cached) as f:
                assert np.array_equal(f['oct'],x[0,0]) and np.array_equal(f['octa'],y[0,0])
                layers=f['layers'].copy()
            source=torch.from_numpy(x).to(device)
            predictions=dict(oct=x[0,0],truth=y[0,0],
                voxel=rollout(voxel,source,8).clamp(0,1)[0,0].cpu().numpy(),
                haar=decode(rollout(haar,encode(source,'haar'),8),'haar').clamp(0,1)[0,0].cpu().numpy(),
                small=small(source).clamp(0,1)[0,0].cpu().numpy(),
                matched=matched(source).clamp(0,1)[0,0].cpu().numpy())
            truth,empty=project(y[0,0],layers)
            arrays={}
            row=dict(id=identity,split=split,case_id=cid,origin_ZYX=[0,loc,loc],
                     cached_input_exact_match=True,cached_sha256=record['sha256'],empty_rays=empty,metrics={})
            for name,a in predictions.items():
                image,_=project(a,layers)
                arrays[name]=image
                if name not in ('oct','truth'):
                    row['metrics'][name]=dict(pcc=pcc(torch.from_numpy(image),torch.from_numpy(truth)),
                        ssim=float(structural_similarity(image,truth,data_range=1.,win_size=7)),
                        mae=float(np.abs(image.astype(np.float64)-truth).mean()))
            np.savez_compressed(output/(identity+'.npz'),**arrays)
            results.append(row)
            print(identity,json.dumps(row['metrics']),flush=True)
    write_json(output/'result.json',dict(complete=True,completed=stamp(),cases=results,
        protocol='First train/validation IDs at two fixed locations. Reuse prior official native-640 MATLAB1-based ILM/OPL layers only after inference. Exactly equal OCT/OCTA arrays verified before use. No training, parameter selection or test patients.'))


if __name__=='__main__':
    main()
