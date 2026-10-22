"""Matched single-patch controls: change only the training-time distribution."""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
PRIOR = Path('/docsystem/niuke/lvyuehang/three_diagnostics_20261021')
CACHE = Path('/docsystem/niuke/lvyuehang/fm3d_validation_20261021')
SEED, UPDATES = 20261021, 2400
ARMS = ('fm_uniform', 'direct_t0', 'fm_half_start')
sys.path.insert(0, str(PRIOR))
from run_diagnostics import maps, module, pcc, rollout, scores, sha256, stamp, write_json
from layer_view import project


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model():
    return module('voxel', 'model').VelocityUNet3D(16, True)


def prepare():
    (ROOT/'inputs').mkdir(exist_ok=False)
    (ROOT/'annotations').mkdir()
    records = json.loads((PRIOR/'input_audit.json').read_text())
    selection = json.loads((CACHE/'data/selection.json').read_text())
    inputs = []
    for original in records:
        origin = Path(original['file'])
        assert sha256(origin) == original['sha256']
        dest = ROOT/'inputs'/origin.name
        shutil.copy2(origin, dest)
        row = dict(original, file=str(dest))
        cached = next((r for r in selection['patches'] if r['case_id']==row['case_id']
                       and r['origin_ZYX']==row['origin_ZYX']), None)
        if cached:
            reference = CACHE/cached['file']
            assert sha256(reference)==cached['sha256']
            with np.load(dest) as current, np.load(reference) as saved:
                assert np.array_equal(current['oct'][0,0], saved['oct'])
                assert np.array_equal(current['octa'][0,0], saved['octa'])
                annotation = ROOT/'annotations'/origin.name
                np.savez_compressed(annotation, layers=saved['layers'], vessel_mask=saved['vessel_mask'])
            row.update(annotation=str(annotation), annotation_sha256=sha256(annotation),
                       cached_input_exact_match=True, cached_reference_sha256=cached['sha256'])
        inputs.append(row)
    torch.manual_seed(SEED)
    model=make_model()
    torch.save(dict(model=model.state_dict(), seed=SEED), ROOT/'initial.pt')
    generator=np.random.default_rng(20261022)
    uniform=generator.uniform(0,1,UPDATES).astype(np.float32)
    mask=np.arange(UPDATES)%2==0
    np.savez_compressed(ROOT/'schedule.npz', fm_uniform=uniform,
                        direct_t0=np.zeros_like(uniform), fm_half_start=np.where(mask,0,uniform),
                        base_uniform=uniform, start_mask=mask)
    dependencies = [PRIOR/'run_diagnostics.py', PRIOR/'layer_view.py', PRIOR/'same_backbone_control.py', PRIOR/'source/voxel/fm3d/model.py',
                    PRIOR/'source/voxel/fm3d/__init__.py']
    protocol = dict(created=stamp(), seed=SEED, updates=UPDATES, arms=ARMS, physical_gpu=0,
        training_patch=inputs[0]['id'], inputs=inputs,
        model='Exact previously frozen voxel VelocityUNet3D(base16, condition=True)',
        parameter_count=sum(p.numel() for p in model.parameters()), initial_state_sha256=state_hash(model),
        initial_file_sha256=sha256(ROOT/'initial.pt'), schedule_sha256=sha256(ROOT/'schedule.npz'),
        optimizer=dict(name='AdamW',lr=2e-4,weight_decay=0,betas=[.9,.999],eps=1e-8,clip_norm=1.),
        precision='FP16 AMP, GradScaler initial1024; FP32 loss and evaluation; TF32 disabled',
        objective='All groups use MSE(v,(OCTA-OCT)) at state=(1-t)*OCT+t*OCTA; only t schedule differs',
        arm_definitions=dict(fm_uniform='2400 uniform times',direct_t0='2400 times t=0; algebraically direct endpoint MSE',
            fm_half_start='Exactly 1200 t=0 updates alternating with 1200 uniform updates; reuse baseline uniform times at corresponding indices'),
        compute_control='One network forward/backward per training attempt; identical successful updates, optimizer and initial weights',
        primary_inference=dict(fm_uniform='Euler8 from OCT',direct_t0='One direct source-only step',fm_half_start='Euler8 from OCT'),
        diagnostic_inference='All arms additionally evaluated with direct1 and Euler8/32. Direct_t0 has no training at t>0; its Euler output is not its intended inference.',
        selection='Fixed final2400, no validation selection; train10301/Y=X=64 only; evaluate original 8 patches; no test patients',
        criteria='Descriptive memorization threshold on primary output: full-Z max PCC>=.90 and SSIM>=.80; also report ILM-OPL separately. Not a statistical test.',
        limitations='One seed, one training patch, two original validation patients. Runtime is shared-GPU time, not a fair throughput benchmark.',
        dependencies={str(p):sha256(p) for p in dependencies},
        environment=dict(pytorch=torch.__version__,numpy=np.__version__,
            gpu_state_at_setup=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True),
            existing_jobs=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()))
    protocol['environment']['existing_jobs']=[line for line in protocol['environment']['existing_jobs']
        if '/conda_envs/' in line and ('fm3d' in line or 'run256.py' in line)]
    write_json(ROOT/'protocol.json',protocol)
    print('PREPARED',json.dumps(dict(initial_state_sha256=state_hash(model),parameters=protocol['parameter_count'],
                                    annotated_patches=sum('annotation' in r for r in inputs))),flush=True)


def load_samples(protocol):
    samples=[]
    for meta in protocol['inputs']:
        assert sha256(meta['file'])==meta['sha256']
        with np.load(meta['file']) as data:
            x,y=(torch.from_numpy(data[key].copy()) for key in ('oct','octa'))
        assert x.shape==y.shape==(1,1,256,64,64)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert 0 <= float(x.min()) <= float(x.max()) <= 1
        assert 0 <= float(y.min()) <= float(y.max()) <= 1
        sample=dict(meta=meta,x=x,y=y)
        if 'annotation' in meta:
            assert sha256(meta['annotation'])==meta['annotation_sha256']
            with np.load(meta['annotation']) as data:
                sample['layers']=data['layers'].copy()
        samples.append(sample)
    assert samples[0]['meta']['id']==protocol['training_patch']
    return samples


def metrics(prediction,sample):
    raw=scores(prediction,sample['y'])
    pred=prediction.detach().float().cpu().clamp(0,1)
    result=scores(pred,sample['y'])
    if 'layers' in sample:
        a,empty=project(pred[0,0].numpy(),sample['layers'])
        b,_=project(sample['y'][0,0].numpy(),sample['layers'])
        result.update(ilm_opl_pcc=pcc(torch.from_numpy(a),torch.from_numpy(b)),
                      ilm_opl_ssim=float(structural_similarity(a,b,data_range=1.,win_size=7)),
                      ilm_opl_mae=float(np.abs(a.astype(np.float64)-b).mean()),empty_rays=empty)
    return dict(raw=raw,final_clamp=result)


@torch.inference_mode()
def evaluate(model,sample,device,steps=(1,8),output=None):
    model.eval()
    source=sample['x'].to(device)
    records={}
    for n in steps:
        prediction=rollout(model,source,n)
        records[str(n)]=metrics(prediction,sample)
        if output is not None:
            maps(output/f'{sample["meta"]["id"]}_euler{n}.npz',prediction.clamp(0,1))
            if 'layers' in sample:
                image,_=project(prediction.clamp(0,1)[0,0].cpu().numpy(),sample['layers'])
                np.savez_compressed(output/f'{sample["meta"]["id"]}_euler{n}_layer.npz',ilm_opl=image)
    teacher=[]
    for value in (0.,.01,.05,.1,.5,.9):
        target=sample['y'].to(device)
        state=(1-value)*source+value*target
        velocity=model(state,torch.tensor([value],device=device),source).float()
        teacher.append(dict(time=value,velocity_mse=F.mse_loss(velocity,target-source).item()))
    return dict(outputs=records,teacher=teacher)


def preflight(device):
    torch.manual_seed(SEED)
    model=make_model().to(device)
    source=torch.rand(1,1,16,16,16,device=device)
    target=torch.rand_like(source)
    t=torch.zeros(1,device=device)
    prediction=source+model(source,t,source)
    direct=F.mse_loss(prediction,target)
    direct_grads=torch.autograd.grad(direct,tuple(model.parameters()))
    endpoint=module('voxel','model').flow_loss(model,source,target,t)
    fm_grads=torch.autograd.grad(endpoint,tuple(model.parameters()))
    max_error=max(float((a-b).abs().max()) for a,b in zip(direct_grads,fm_grads))
    assert abs(float(direct.detach())-float(endpoint.detach()))<1e-6 and max_error<1e-5
    with np.load(ROOT/'schedule.npz') as schedule:
        assert len(schedule['fm_uniform'])==UPDATES
        assert np.count_nonzero(schedule['direct_t0'])==0
        assert np.count_nonzero(schedule['fm_half_start']==0)==UPDATES//2
        assert np.array_equal(schedule['fm_half_start'][~schedule['start_mask']],schedule['fm_uniform'][~schedule['start_mask']])
    write_json(ROOT/'preflight.json',dict(passed=True,source_only_direct_mse_equals_t0_flow_mse=True,
               direct_loss=float(direct.detach()),t0_loss=float(endpoint.detach()),max_gradient_error=max_error,
               schedule_counts_and_coupled_uniform_times_verified=True))
    del model
    torch.cuda.empty_cache()


def train_arm(arm,protocol,samples,device):
    output=ROOT/arm
    output.mkdir(exist_ok=False)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model=make_model().to(device)
    model.load_state_dict(torch.load(ROOT/'initial.pt',map_location='cpu',weights_only=False)['model'])
    initial_hash=state_hash(model)
    assert initial_hash==protocol['initial_state_sha256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    with np.load(ROOT/'schedule.npz') as data:
        times=data[arm].copy()
    x,y=(samples[0][key].to(device) for key in ('x','y'))
    flow_loss=module('voxel','model').flow_loss
    history=[]
    completed=skipped=attempts=0
    tick=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    primary_steps=1 if arm=='direct_t0' else 8

    def measure():
        report=evaluate(model,samples[0],device)
        record=dict(update=completed,seconds=time.perf_counter()-tick,**report)
        history.append(record)
        write_json(output/'history.json',history)
        print(arm,completed,'primary',json.dumps(report['outputs'][str(primary_steps)]['final_clamp']),
              'seconds',round(record['seconds'],1),flush=True)
        write_json(ROOT/'status.json',dict(state='running',arm=arm,update=completed,target=UPDATES,updated=stamp()))

    measure()
    with (output/'steps.jsonl').open('w',encoding='utf-8') as trace:
        while completed<UPDATES:
            value=float(times[completed])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=True):
                loss=flow_loss(model,x,y,torch.tensor([value],device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
            old_scale=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempts+=1
            if scaler.get_scale()<old_scale:
                skipped+=1
                if skipped>100:
                    raise FloatingPointError('Repeated AMP overflow')
                continue
            if not math.isfinite(float(grad_norm)) or not float(grad_norm)>0:
                raise FloatingPointError('No finite nonzero gradient')
            completed+=1
            trace.write(json.dumps(dict(update=completed,time=value,loss=float(loss.detach()),
                                        gradient_norm=float(grad_norm)))+'\n')
            if completed%50==0:
                trace.flush()
                write_json(ROOT/'status.json',dict(state='running',arm=arm,update=completed,target=UPDATES,updated=stamp()))
            if completed%200==0:
                measure()
            if completed%800==0:
                payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                             update=completed,arm=arm,initial_state_sha256=initial_hash,protocol=protocol)
                temporary=output/f'update_{completed:04d}.tmp'
                torch.save(payload,temporary)
                temporary.replace(output/f'update_{completed:04d}.pt')
    evaluation=[]
    with torch.inference_mode():
        for sample in samples:
            report=evaluate(model,sample,device,(1,8,32),output/'maps')
            primary=rollout(model,sample['x'].to(device),primary_steps).clamp(0,1).cpu()
            (output/'predictions').mkdir(exist_ok=True)
            np.savez_compressed(output/'predictions'/(sample['meta']['id']+'.npz'),
                                octa=primary[0,0].numpy(),octa_gt=sample['y'][0,0].numpy(),axis_order=np.array('ZYX'))
            evaluation.append(dict(**sample['meta'],trained_on_this_patch=sample is samples[0],**report))
    write_json(output/'result.json',dict(complete=True,arm=arm,successful_updates=completed,
        attempts=attempts,amp_skipped_updates=skipped,training_forward_calls=attempts,
        initial_state_sha256=initial_hash,final_state_sha256=state_hash(model),
        training_patch=protocol['training_patch'],initial_file_sha256=protocol['initial_file_sha256'],
        schedule_sha256=protocol['schedule_sha256'],zero_time_updates=int(np.count_nonzero(times==0)),
        elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        primary_steps=primary_steps,evaluation=evaluation))
    del model,optimizer,scaler
    torch.cuda.empty_cache()


def final_audit(protocol):
    audits={}
    for arm in ARMS:
        report=json.loads((ROOT/arm/'result.json').read_text())
        with np.load(ROOT/'schedule.npz') as schedule:
            expected=schedule[arm]
        trace=[json.loads(line) for line in (ROOT/arm/'steps.jsonl').read_text().splitlines()]
        assert len(trace)==UPDATES and [r['update'] for r in trace]==list(range(1,UPDATES+1))
        assert np.array_equal(np.array([r['time'] for r in trace],dtype=np.float32),expected)
        assert report['initial_state_sha256']==protocol['initial_state_sha256']
        for row in report['evaluation']:
            with np.load(ROOT/arm/'predictions'/(row['id']+'.npz')) as prediction, np.load(row['file']) as original:
                assert np.array_equal(prediction['octa_gt'],original['octa'][0,0])
                assert np.isfinite(prediction['octa']).all()
                mse=float(np.square(prediction['octa'].astype(np.float64)-prediction['octa_gt']).mean())
                assert abs(mse-row['outputs'][str(report['primary_steps'])]['final_clamp']['mse'])<1e-10
        audits[arm]=dict(successful_updates=report['successful_updates'],zero_time_updates=report['zero_time_updates'],
                        amp_skipped_updates=report['amp_skipped_updates'],schedule_and_saved_truth_verified=True)
    unchanged=all(sha256(path)==digest for path,digest in protocol['dependencies'].items())
    assert unchanged and sha256(ROOT/'initial.pt')==protocol['initial_file_sha256']
    assert sha256(ROOT/'schedule.npz')==protocol['schedule_sha256']
    write_json(ROOT/'final_audit.json',dict(passed=True,completed=stamp(),arms=audits,
        same_initial_parameters=True,same_training_patch=True,same_successful_updates=True,
        fixed_schedule_replayed=True,frozen_dependencies_unchanged=unchanged,no_test_patients=True,
        saved_3d_predictions_verified=24))
    write_json(ROOT/'status.json',dict(state='complete',completed=stamp(),arms=3,successful_updates=3*UPDATES))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.phase=='prepare':
        prepare()
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    protocol=json.loads((ROOT/'protocol.json').read_text())
    write_json(ROOT/'status.json',dict(state='running',stage='preflight',updated=stamp()))
    samples=load_samples(protocol)
    preflight(torch.device('cuda:0'))
    for sample in samples:
        maps(ROOT/'reference_maps'/(sample['meta']['id']+'_oct.npz'),sample['x'])
        maps(ROOT/'reference_maps'/(sample['meta']['id']+'_octa.npz'),sample['y'])
        if 'layers' in sample:
            a,_=project(sample['x'][0,0].numpy(),sample['layers'])
            b,_=project(sample['y'][0,0].numpy(),sample['layers'])
            np.savez_compressed(ROOT/'reference_maps'/(sample['meta']['id']+'_layers.npz'),oct=a,octa=b)
    for arm in ARMS:
        train_arm(arm,protocol,samples,torch.device('cuda:0'))
    final_audit(protocol)
    print('MATCHED THREE-ARM CONTROL COMPLETE',stamp(),flush=True)


if __name__=='__main__':
    main()
