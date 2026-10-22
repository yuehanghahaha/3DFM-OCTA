"""Continue all matched arms to equal4800/equal7200 without altering the first study."""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
BASE=Path('/docsystem/niuke/lvyuehang/fm_start_control_20261022')
sys.path.insert(0,str(BASE))
from run_control import (ARMS,evaluate,load_samples,make_model,module,rollout,sha256,
                         stamp,state_hash,write_json)

STAGES=(4800,7200)


def prepare():
    (ROOT/'seeds').mkdir(exist_ok=False)
    original=json.loads((BASE/'protocol.json').read_text())
    assert json.loads((BASE/'final_audit.json').read_text())['passed']
    assert json.loads((BASE/'verification.json').read_text())['passed']
    rng=np.random.default_rng(20261022)
    uniform=rng.uniform(0,1,STAGES[-1]).astype(np.float32)
    mask=np.arange(STAGES[-1])%2==0
    arrays=dict(fm_uniform=uniform,direct_t0=np.zeros_like(uniform),
                fm_half_start=np.where(mask,0,uniform),base_uniform=uniform,start_mask=mask)
    with np.load(BASE/'schedule.npz') as old:
        assert all(np.array_equal(value[:2400],old[name]) for name,value in arrays.items())
    np.savez_compressed(ROOT/'schedule.npz',**arrays)
    resumes={}
    for arm in ARMS:
        source=BASE/arm/'update_2400.pt'
        dest=ROOT/'seeds'/(arm+'.pt')
        shutil.copy2(source,dest)
        payload=torch.load(dest,map_location='cpu',weights_only=False)
        result=json.loads((BASE/arm/'result.json').read_text())
        assert payload['update']==result['successful_updates']==2400
        model=make_model()
        model.load_state_dict(payload['model'])
        assert state_hash(model)==result['final_state_sha256']
        assert all(float(state['step'])==2400 for state in payload['optimizer']['state'].values())
        resumes[arm]=dict(checkpoint=str(dest),sha256=sha256(dest),original_sha256=sha256(source),
            state_sha256=state_hash(model),optimizer_step=2400,scaler_state=payload['scaler'])
    dependencies=dict(original['dependencies'])
    dependencies.update({str(BASE/name):sha256(BASE/name) for name in
        ('run_control.py','protocol.json','schedule.npz','final_audit.json','verification.json')})
    protocol=dict(original,created=stamp(),updates=STAGES[-1],start_update=2400,stages=list(STAGES),
        original_study=str(BASE),resume=resumes,schedule_sha256=sha256(ROOT/'schedule.npz'),
        original_schedule_prefix_verified=True,dependencies=dependencies,
        training='Resume model, AdamW state and AMP scaler; no LR changes, resets or extra objectives',
        selection='Compare fixed2400/4800/7200 for all arms; no validation selection or early stopping',
        run_script_sha256=sha256(Path(__file__)))
    write_json(ROOT/'protocol.json',protocol)
    print('PREPARED: same schedule prefix, checkpoint weights and AdamW step2400 verified',flush=True)


def run_stage(arm,end,protocol,samples,device):
    start=end-2400
    source=ROOT/'seeds'/(arm+'.pt') if start==2400 else ROOT/f'stage_{start}'/arm/f'update_{start:04d}.pt'
    payload=torch.load(source,map_location='cpu',weights_only=False)
    assert payload['update']==start
    output=ROOT/f'stage_{end}'/arm
    output.mkdir(parents=True,exist_ok=False)
    model=make_model().to(device)
    model.load_state_dict(payload['model'])
    loaded_hash=state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0)
    optimizer.load_state_dict(payload['optimizer'])
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    scaler.load_state_dict(payload['scaler'])
    assert all(group['lr']==2e-4 and group['weight_decay']==0 for group in optimizer.param_groups)
    assert all(float(state['step'])==start for state in optimizer.state.values())
    del payload
    with np.load(ROOT/'schedule.npz') as schedule:
        times=schedule[arm].copy()
    x,y=(samples[0][key].to(device) for key in ('x','y'))
    flow_loss=module('voxel','model').flow_loss
    completed,skipped,attempts=start,0,0
    history=[]
    primary_steps=1 if arm=='direct_t0' else 8
    tick=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def measure():
        report=evaluate(model,samples[0],device)
        history.append(dict(update=completed,seconds=time.perf_counter()-tick,**report))
        write_json(output/'history.json',history)
        print('STAGE',end,arm,completed,'primary',json.dumps(report['outputs'][str(primary_steps)]['final_clamp']),flush=True)
        write_json(ROOT/'status.json',dict(state='running',stage=end,arm=arm,update=completed,target=end,updated=stamp()))

    measure()
    reference=json.loads((BASE/arm/'result.json').read_text()) if start==2400 else json.loads((ROOT/f'stage_{start}'/arm/'result.json').read_text())
    old_metric=reference['evaluation'][0]['outputs'][str(primary_steps)]['final_clamp']
    resumed=history[0]['outputs'][str(primary_steps)]['final_clamp']
    max_resume_error=max(abs(resumed[key]-old_metric[key]) for key in ('mae','mse','ilm_opl_pcc','ilm_opl_ssim'))
    assert max_resume_error<1e-6
    with (output/'steps.jsonl').open('w',encoding='utf-8') as trace:
        while completed<end:
            value=float(times[completed])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=True):
                loss=flow_loss(model,x,y,torch.tensor([value],device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
            previous_scale=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempts+=1
            if scaler.get_scale()<previous_scale:
                skipped+=1
                if skipped>100:
                    raise FloatingPointError('Repeated AMP overflow')
                continue
            if not math.isfinite(float(grad_norm)) or not float(grad_norm)>0:
                raise FloatingPointError('Invalid gradient')
            completed+=1
            trace.write(json.dumps(dict(update=completed,time=value,loss=float(loss.detach()),gradient_norm=float(grad_norm)))+'\n')
            if completed%100==0:
                trace.flush()
                write_json(ROOT/'status.json',dict(state='running',stage=end,arm=arm,update=completed,target=end,updated=stamp()))
            if completed%400==0:
                measure()
            if completed%1200==0:
                checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                                update=completed,arm=arm,initial_state_sha256=protocol['initial_state_sha256'])
                temporary=output/f'update_{completed:04d}.tmp'
                torch.save(checkpoint,temporary)
                temporary.replace(output/f'update_{completed:04d}.pt')
    evaluation=[]
    with torch.inference_mode():
        for sample in samples:
            report=evaluate(model,sample,device,(1,8,32),output/'maps')
            prediction=rollout(model,sample['x'].to(device),primary_steps).clamp(0,1).cpu()
            (output/'predictions').mkdir(exist_ok=True)
            np.savez_compressed(output/'predictions'/(sample['meta']['id']+'.npz'),
                octa=prediction[0,0].numpy(),octa_gt=sample['y'][0,0].numpy(),axis_order=np.array('ZYX'))
            evaluation.append(dict(**sample['meta'],trained_on_this_patch=sample is samples[0],**report))
    assert all(float(state['step'])==end for state in optimizer.state.values())
    write_json(output/'result.json',dict(complete=True,arm=arm,start_update=start,successful_updates=end,
        additional_successful_updates=end-start,attempts=attempts,amp_skipped_updates=skipped,
        loaded_checkpoint=str(source),loaded_checkpoint_sha256=sha256(source),loaded_state_sha256=loaded_hash,
        max_resumed_metric_error=max_resume_error,optimizer_restored_step=start,optimizer_final_step=end,
        final_state_sha256=state_hash(model),initial_state_sha256=protocol['initial_state_sha256'],
        training_patch=protocol['training_patch'],schedule_sha256=protocol['schedule_sha256'],
        zero_time_updates_total=int(np.count_nonzero(times[:end]==0)),primary_steps=primary_steps,
        elapsed_seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),evaluation=evaluation))
    del model,optimizer,scaler,checkpoint
    torch.cuda.empty_cache()


def audit(protocol):
    audits=[]
    with np.load(ROOT/'schedule.npz') as schedules:
        for end in STAGES:
            for arm in ARMS:
                output=ROOT/f'stage_{end}'/arm
                result=json.loads((output/'result.json').read_text())
                records=[json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
                assert [r['update'] for r in records]==list(range(end-2399,end+1))
                assert np.array_equal(np.array([r['time'] for r in records],np.float32),schedules[arm][end-2400:end])
                assert result['successful_updates']==end and result['optimizer_final_step']==end
                for row in result['evaluation']:
                    with np.load(output/'predictions'/(row['id']+'.npz')) as prediction,np.load(row['file']) as original:
                        truth=original['octa'][0,0]
                        assert np.array_equal(prediction['octa_gt'],truth)
                        assert prediction['octa'].shape==truth.shape and np.isfinite(prediction['octa']).all()
                        mse=float(np.square(prediction['octa'].astype(np.float64)-truth).mean())
                        assert abs(mse-row['outputs'][str(result['primary_steps'])]['final_clamp']['mse'])<1e-10
                audits.append(dict(arm=arm,budget=end,additional_updates=len(records),passed=True))
    unchanged=all(sha256(path)==digest for path,digest in protocol['dependencies'].items())
    assert unchanged and sha256(Path(__file__))==protocol['run_script_sha256']
    assert sha256(ROOT/'schedule.npz')==protocol['schedule_sha256']
    assert all(sha256(ROOT/'seeds'/(arm+'.pt'))==r['sha256'] for arm,r in protocol['resume'].items())
    write_json(ROOT/'final_audit.json',dict(passed=True,completed=stamp(),records=audits,
        optimizer_and_scaler_resumed=True,schedule_prefix_and_continuation_verified=True,
        original_study_and_dependencies_unchanged=True,saved_3d_predictions_verified=48,
        additional_successful_updates=14400,total_successful_updates_including_original=21600))
    write_json(ROOT/'status.json',dict(state='complete',completed=stamp(),final_updates_per_arm=7200,stages=list(STAGES)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.phase=='prepare':
        prepare()
        return
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    protocol=json.loads((ROOT/'protocol.json').read_text())
    samples=load_samples(protocol)
    for end in STAGES:
        for arm in ARMS:
            run_stage(arm,end,protocol,samples,torch.device('cuda:0'))
        write_json(ROOT/f'completed_{end}.json',dict(complete=True,budget=end,completed=stamp()))
    audit(protocol)
    print('ALL ARMS REACHED7200',stamp(),flush=True)


if __name__=='__main__':
    main()
