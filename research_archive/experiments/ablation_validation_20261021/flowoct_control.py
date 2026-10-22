"""Public-code FlowOCT on exactly the frozen 3D validation tensors."""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from flowoct.model import FlowOCT
import monai
import numpy as np
import torch
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from engine import (aggregate,load_samples,panel,sample_metrics,save_checkpoint,
                    seed_all,sha256,write_json)
from run import stage_samples


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def to_flowoct(volume):
    """[B,C,Z,Y,X] -> author loader's [B,C,Y,X,Z], no interpolation/scaling."""
    return volume.permute(0,1,3,4,2).contiguous()


def from_flowoct(volume):
    return volume.permute(0,1,4,2,3).contiguous()


def make_flowoct(config,device):
    opt=SimpleNamespace(gpu_ids=[device.index or 0] if device.type=='cuda' else [],
        isTrain=True,checkpoints_dir=str(ROOT/'06_flowoct'),name='control',
        lr=config['lr'],beta=config['beta'],train_slices=config['train_slices'],
        network_chunk=config['network_chunk'])
    return FlowOCT(opt)


@torch.inference_mode()
def predict(flow,source,mode='shared8',chunk_size=16):
    """Only OCT enters either generation path; targets are evaluated outside."""
    native=to_flowoct(source)
    if mode=='native3':
        return from_flowoct(flow.validation(native,None,chunk_size)).float()
    if mode!='shared8':
        raise ValueError(mode)
    b,c,d,h,w=native.shape
    flat=native.permute(0,2,1,3,4).reshape(-1,c,h,w)
    outputs=[]
    for start in range(0,len(flat),chunk_size):
        state=flat[start:start+chunk_size]
        for index in range(8):
            t0=torch.full((len(state),),index/8,device=source.device)
            t1=torch.full_like(t0,(index+1)/8)
            state=flow.euler_step(state,t0,t1)
        outputs.append(state.clamp(0,1))
    native=torch.cat(outputs).reshape(b,d,c,h,w).permute(0,2,1,3,4)
    return from_flowoct(native).float()


@torch.inference_mode()
def evaluate_predictor(predictor,samples,destination=None):
    records=[]
    for sample in samples:
        prediction=predictor(sample['oct'])
        if isinstance(prediction,torch.Tensor):
            prediction=prediction.detach().float().cpu().numpy().squeeze((0,1))
        if prediction.shape!=sample['octa'].shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid generated shape/values')
        if prediction.min()<0 or prediction.max()>1:
            raise ValueError('Predictor must apply its declared final range handling')
        metrics,a,b=sample_metrics(prediction,sample)
        row={k:sample[k] for k in ('case_id','patient_id','split','disease','patch_id')}
        row['metrics']=metrics
        records.append(row)
        if destination is not None:
            destination.mkdir(parents=True,exist_ok=True)
            name=sample['case_id']+'_'+sample['patch_id']
            np.savez_compressed(destination/(name+'.npz'),octa=prediction,octa_gt=sample['octa'],axis_order=np.array('ZYX'))
            panel(sample,prediction,a,b,destination/(name+'.png'))
    return dict(cases=records,summary=aggregate(records))


def evaluate_flow(flow,samples,mode,device,destination=None):
    previous=flow.training
    flow.eval()
    try:
        return evaluate_predictor(lambda source:predict(flow,torch.from_numpy(source)[None,None].to(device),mode),samples,destination)
    finally:
        flow.train(previous)


def fingerprint(config,suite,stage):
    files=[ROOT/'flowoct_control.py',ROOT/'engine.py']+sorted((ROOT/'source/flowoct').glob('*.py'))
    signature=dict(config=config,suite=suite,stage=stage,
        selection_sha256=sha256(ROOT/'data/selection.json'),
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in files})
    return signature,hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()


def train_stage(config,suite,selection,samples,stage,device,resume):
    output=ROOT/'06_flowoct'/stage['name']
    output.mkdir(parents=True,exist_ok=True)
    evaluation,training=stage_samples(samples,stage['name'],selection)
    reference=ROOT/'01_fm3d'/stage['name']/'schedule.npz'
    with np.load(reference,allow_pickle=False) as f:
        indices=f['sample_indices'].copy()
    assert len(indices)==stage['updates']
    signature,signature_hash=fingerprint(config,suite,stage)
    seed_all(config['seed'])
    flow=make_flowoct(config,device)
    completed,best=0,-float('inf')
    latest=output/'latest.pt'
    if latest.exists():
        if not resume:
            raise FileExistsError('Existing FlowOCT run; use --resume')
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['signature_sha256']!=signature_hash:
            raise ValueError('FlowOCT protocol/data/source changed')
        flow.load_state_dict(saved['model'])
        flow.optimizer_G.load_state_dict(saved['optimizer'])
        completed,best=saved['completed_updates'],saved['best_ssim']
        del saved
        if completed==stage['updates'] and (output/'final_shared8.json').exists() and (output/'final_native3.json').exists():
            print('Already completed',stage['name'],flush=True)
            return
    elif any(output.iterdir()):
        raise FileExistsError('Nonempty FlowOCT stage without checkpoint')
    write_json(output/'protocol.json',signature)
    write_json(output/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,monai=monai.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),generator_parameters=sum(p.numel() for p in flow.netG.parameters()),
        contrast_head_parameters=sum(p.numel() for p in flow.contrast_projection_head.parameters()),
        generator_defaults=str(inspect.signature(DiffusionModelUNet)),input_voxels_ZYX=[256,64,64],
        model_tensor_YXZ=[64,64,256],sampled_Bscans_per_update=16,amp=False,gradient_clipping=False,
        comparison='Same tensors/case order/successful updates. Native 2D slice/time sampling, Adam and losses differ by method.'))
    np.savez(output/'schedule.npz',sample_indices=indices)
    if not (output/'initial_shared8.json').exists():
        write_json(output/'initial_shared8.json',evaluate_flow(flow,evaluation,'shared8',device))
    flow.train()
    tick=time.perf_counter()
    losses=[]
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for update in range(completed+1,stage['updates']+1):
        sample=training[int(indices[update-1])]
        # The author's forward samples its own B-scans and one t per B-scan.
        # Seeding each update makes recovery reproducible without changing it.
        update_seed=stage['seed']*10000+update
        seed_all(update_seed)
        source,target=[torch.from_numpy(sample[k])[None,None].to(device) for k in ('oct','octa')]
        flow.set_input(dict(A=to_flowoct(source),B=to_flowoct(target)))
        flow.forward()  # Author's four losses, backward and Adam step, all FP32.
        row=dict(flow.get_current_loss())
        row.update(loss=float(flow.loss.detach()),update=update,case_id=sample['case_id'],patch_id=sample['patch_id'],seed=update_seed)
        expected=10*row['vloss']+.5*row['L1loss']+.005*row['contrastloss']+.1*row['proj_loss']
        if abs(row['loss']-expected)>1e-6*max(1,abs(expected)):
            raise ValueError('FlowOCT loss assembly differs')
        with (output/'steps.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        losses.append(row)
        if update%25==0:
            write_json(ROOT/'06_flowoct/status.json',dict(status='running',stage=stage['name'],updates=update,
                target_updates=stage['updates'],seconds=time.perf_counter()-tick,last_loss=row['loss']))
            print('FlowOCT',stage['name'],update,stage['updates'],'loss',row['loss'],flush=True)
        if update%config['eval_every']==0 or update==stage['updates']:
            report=evaluate_flow(flow,evaluation,'shared8',device)
            write_json(output/f'metrics_{update:05d}.json',report)
            score=report['summary']['validation']['metrics']['ilm_opl_ssim']
            improved=score>best
            best=max(best,score)
            payload=dict(model=flow.state_dict(),optimizer=flow.optimizer_G.state_dict(),completed_updates=update,
                best_ssim=best,signature_sha256=signature_hash,protocol=signature)
            save_checkpoint(latest,payload)
            if improved:
                save_checkpoint(output/'best_validation.pt',payload)
            history=dict(update=update,seconds=time.perf_counter()-tick,summary=report['summary'],
                mean_loss={key:float(np.mean([r[key] for r in losses])) for key in ('loss','vloss','L1loss','contrastloss','proj_loss')},
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            with (output/'history.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(history,allow_nan=False)+'\n')
            losses=[]
            print('FlowOCT evaluation',stage['name'],update,report['summary'],flush=True)
    for mode in ('native3','shared8'):
        report=evaluate_flow(flow,evaluation,mode,device,output/('predictions_'+mode))
        report.update(arm='06_flowoct',stage=stage['name'],completed_updates=stage['updates'],solver=mode)
        write_json(output/('final_'+mode+'.json'),report)
    write_json(output/'status.json',dict(status='complete',updates=stage['updates']))
    del flow
    if device.type=='cuda':
        torch.cuda.empty_cache()


def benchmark(config,samples,device):
    seed_all(config['seed'])
    flow=make_flowoct(config,device)
    sample=samples[0]
    source,target=[torch.from_numpy(sample[k])[None,None].to(device) for k in ('oct','octa')]
    flow.set_input(dict(A=to_flowoct(source),B=to_flowoct(target)))
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tick=time.perf_counter()
    for _ in range(3):
        flow.forward()
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    record=dict(seconds_per_update=(time.perf_counter()-tick)/3,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        generator_parameters=sum(p.numel() for p in flow.netG.parameters()),losses=dict(flow.get_current_loss()),
        tensor_YXZ=list(flow.real_A.shape),sampled_slices=flow.real_A.shape[2]//4,
        full_batch=True,network_chunk=0,amp=False)
    write_json(ROOT/'06_flowoct/benchmark.json',record)
    print(json.dumps(record,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--stage',choices=['all','one_case','few_case'],default='all')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=torch.device(args.device)
    config,suite,selection=read(ROOT/'06_flowoct/config.json'),read(ROOT/'suite.json'),read(ROOT/'data/selection.json')
    samples=load_samples(ROOT,selection)
    if args.benchmark:
        benchmark(config,samples,device)
        return
    for stage in suite['stages']:
        if args.stage in ('all',stage['name']):
            train_stage(config,suite,selection,samples,stage,device,args.resume)
    write_json(ROOT/'06_flowoct/status.json',dict(status='complete',stages=args.stage))


if __name__=='__main__':
    main()
