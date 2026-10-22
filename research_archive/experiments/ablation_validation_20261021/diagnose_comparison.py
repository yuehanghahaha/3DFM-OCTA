"""Separate time-scale/starting-point controls and teacher-state diagnostics."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
import torch
from torch.nn import functional as F
from engine import (encode,decode,load_samples,make_model,objectives,rollout,
                    sample_metrics,save_checkpoint,seed_all,sha256,write_json)
from run import stage_samples
from flowoct_control import (evaluate_predictor,make_flowoct,read,to_flowoct,
                             from_flowoct,predict)


def tensors(sample,device):
    return [torch.from_numpy(sample[k])[None,None].to(device) for k in ('oct','octa')]


@torch.inference_mode()
def predict_control(model,source,device,kind,amp=True):
    source=torch.from_numpy(source)[None,None].to(device)
    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
        if kind=='start_only_mse':
            output=source+model(source,torch.zeros(1,device=device),source).float()
        else:
            output=rollout(model,source,'voxel',8)
    return output.float().clamp(0,1)


def controls(samples,selection,suite,device):
    evaluation,training=stage_samples(samples,'one_case',selection)
    assert len(training)==1
    sample=training[0]
    source,target=tensors(sample,device)
    with np.load(ROOT/'01_fm3d/one_case/schedule.npz',allow_pickle=False) as f:
        reference_times=f['times'].copy()
    for kind,updates in (('time_scale_1',400),('start_only_mse',1600)):
        output=ROOT/'diagnostics'/kind
        if (output/'final_metrics.json').exists():
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Use a fresh diagnostic run; interrupted controls are not silently overwritten')
        output.mkdir(parents=True,exist_ok=True)
        seed_all(suite['seed'])
        model=make_model(suite['base_channels'],suite['condition'],'voxel').to(device)
        parameter_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        if kind=='time_scale_1':
            model.frequencies.div_(1000.)
        config=dict(control=kind,updates=updates,case_id=sample['case_id'],patch_id=sample['patch_id'],
            initial_parameter_sha256=parameter_hash,seed=suite['seed'],
            time_scale=1 if kind=='time_scale_1' else 1000,
            target='Same residual MSE as baseline; t=0 on every update for start_only_mse',
            time_sequence='Exact saved baseline 400-value sequence' if kind=='time_scale_1' else 'All zero',
            prediction='Free Euler8' if kind=='time_scale_1' else 'OCT+v(OCT,0,OCT); direct capacity control, not a trained ODE',
            selection_sha256=sha256(ROOT/'data/selection.json'),lr=suite['lr'],amp=suite['amp'])
        write_json(output/'config.json',config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=suite['lr'],weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024,enabled=suite['amp'] and device.type=='cuda')
        def evaluate(destination=None):
            previous=model.training
            model.eval()
            try:
                return evaluate_predictor(lambda x:predict_control(model,x,device,kind,suite['amp']),evaluation,destination)
            finally:
                model.train(previous)
        write_json(output/'initial_metrics.json',evaluate())
        completed,skipped=0,0
        tick=time.perf_counter()
        while completed<updates:
            t=0. if kind=='start_only_mse' else float(reference_times[completed])
            time_value=torch.tensor([t],device=device)
            optimizer.zero_grad(set_to_none=True)
            arm=dict(representation='voxel',lambda_projection=0.,lambda_reconstruction=0.)
            with torch.autocast(device_type=device.type,enabled=suite['amp'] and device.type=='cuda'):
                loss,_=objectives(model,source,target,time_value,arm,8)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite diagnostic loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
            old_scale=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()<old_scale:
                skipped+=1
                if skipped>100:
                    raise FloatingPointError('Repeated overflow')
                continue
            completed+=1
            with (output/'steps.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(update=completed,time=t,loss=float(loss.detach())))+'\n')
            if completed%100==0:
                write_json(ROOT/'diagnostics/status.json',dict(status='running',control=kind,updates=completed,target_updates=updates))
                print(kind,completed,updates,float(loss.detach()),flush=True)
            if completed%200==0:
                report=evaluate()
                write_json(output/f'metrics_{completed:05d}.json',report)
                history=dict(update=completed,seconds=time.perf_counter()-tick,summary=report['summary'])
                with (output/'history.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(history,allow_nan=False)+'\n')
                print('Control evaluation',kind,completed,report['summary'],flush=True)
        final=evaluate(output/'predictions')
        final.update(control=kind,updates=completed,amp_skipped=skipped,config=config)
        write_json(output/'final_metrics.json',final)
        save_checkpoint(output/'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),completed_updates=completed,config=config))
        del model,optimizer,scaler
        if device.type=='cuda':
            torch.cuda.empty_cache()
    write_json(ROOT/'diagnostics/status.json',dict(status='complete',controls=['time_scale_1','start_only_mse']))


@torch.inference_mode()
def probe_3d(model,source,target,sample,representation,amp):
    records=[]
    c,y=encode(source,representation),encode(target,representation)
    for value in (0.,.001,.01,.05,.5):
        t=torch.tensor([value],device=source.device)
        state=(1-value)*c+value*y
        with torch.autocast(device_type=source.device.type,enabled=amp and source.device.type=='cuda'):
            velocity=model(state,t,c)
        endpoint=decode(state+(1-value)*velocity.float(),representation)
        metrics,_,_=sample_metrics(endpoint.clamp(0,1)[0,0].cpu().numpy(),sample)
        records.append(dict(mode='teacher_state',time=value,
            velocity_mse=float(F.mse_loss(velocity.float(),(y-c).float())),metrics=metrics,
            target_in_network_state=(value>0)))
    for steps in (1,3,8,32):
        with torch.autocast(device_type=source.device.type,enabled=amp and source.device.type=='cuda'):
            endpoint=rollout(model,source,representation,steps)
        metrics,_,_=sample_metrics(endpoint.clamp(0,1)[0,0].cpu().numpy(),sample)
        records.append(dict(mode='OCT_only_rollout',steps=steps,metrics=metrics,target_in_network_state=False))
    return records


@torch.inference_mode()
def probe_flow(flow,source,target,sample):
    native_c,native_y=to_flowoct(source),to_flowoct(target)
    b,c,d,h,w=native_c.shape
    cflat=native_c.permute(0,2,1,3,4).reshape(-1,c,h,w)
    yflat=native_y.permute(0,2,1,3,4).reshape(-1,c,h,w)
    records=[]
    for value in (0.,.001,.01,.05,.5):
        state=(1-value)*cflat+value*yflat
        outputs=[]
        for start in range(0,len(state),16):
            current=state[start:start+16]
            t=torch.full((len(current),),value,device=source.device)
            outputs.append(flow.euler_step(current,t,torch.ones_like(t)))
        endpoint=torch.cat(outputs)
        velocity=(endpoint-state)/(1-value)
        native=endpoint.reshape(b,d,c,h,w).permute(0,2,1,3,4)
        result=from_flowoct(native).clamp(0,1)[0,0].cpu().numpy()
        metrics,_,_=sample_metrics(result,sample)
        records.append(dict(mode='teacher_state',time=value,
            velocity_mse=float(F.mse_loss(velocity,yflat-cflat)),metrics=metrics,target_in_network_state=(value>0)))
    for mode in ('native3','shared8'):
        result=predict(flow,source,mode)[0,0].cpu().numpy()
        metrics,_,_=sample_metrics(result,sample)
        records.append(dict(mode='OCT_only_'+mode,metrics=metrics,target_in_network_state=False))
    return records


def probe(samples,suite,device):
    chosen=[s for s in samples if s['case_id'] in ('10301','10319') and s['patch_id']=='y064_x064']
    output=ROOT/'diagnostics/teacher_probe.json'
    result=dict(scope='Mechanism diagnostic only. t>0 states contain real target signal and are NOT valid inference.',records=[])
    for stage in ('one_case','few_case'):
        for arm,representation in (('01_fm3d','voxel'),('05_fm3d_wavelet_zproj_reconstruction','haar3d')):
            ck=torch.load(ROOT/arm/stage/'latest.pt',map_location='cpu',weights_only=False)
            model=make_model(16,True,representation).to(device)
            model.load_state_dict(ck['model'])
            model.eval()
            del ck
            for sample in chosen:
                source,target=tensors(sample,device)
                result['records'].append(dict(model=arm,stage=stage,case_id=sample['case_id'],split=sample['split'],
                    probes=probe_3d(model,source,target,sample,representation,suite['amp'])))
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
        ck=torch.load(ROOT/'06_flowoct'/stage/'latest.pt',map_location='cpu',weights_only=False)
        flow=make_flowoct(read(ROOT/'06_flowoct/config.json'),device)
        flow.load_state_dict(ck['model'])
        flow.eval()
        del ck
        for sample in chosen:
            source,target=tensors(sample,device)
            result['records'].append(dict(model='06_flowoct',stage=stage,case_id=sample['case_id'],split=sample['split'],
                probes=probe_flow(flow,source,target,sample)))
        del flow
        if device.type=='cuda': torch.cuda.empty_cache()
        write_json(output,result)
    print('Wrote teacher-state and OCT-only generation probes',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda')
    p.add_argument('--action',choices=['controls','probe','all'],default='all')
    a=p.parse_args()
    torch.set_num_threads(2)
    suite,selection=read(ROOT/'suite.json'),read(ROOT/'data/selection.json')
    samples=load_samples(ROOT,selection)
    device=torch.device(a.device)
    if a.action in ('controls','all'): controls(samples,selection,suite,device)
    if a.action in ('probe','all'): probe(samples,suite,device)


if __name__=='__main__':
    main()
