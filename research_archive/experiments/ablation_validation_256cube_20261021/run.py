"""Run five independent, equal-update 3D FM memorization/validation experiments."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))

import numpy as np
import torch
from engine import (ARMS, evaluate, load_samples, make_model, objectives, save_checkpoint,
                    schedule, seed_all, sha256, write_json)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_fingerprint():
    files = [ROOT/name for name in ('engine.py','run.py','prepare.py')]
    files += sorted((ROOT/'source/fm3d').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in files}


def stage_samples(samples, stage, selection):
    if stage == 'few_case':
        return samples, [s for s in samples if s['split'] == 'train']
    case_patch = selection['one_case_patch']
    training = [s for s in samples if s['case_id']+'_'+s['patch_id'] == case_patch]
    return training + [s for s in samples if s['split'] == 'validation'], training


def benchmark(samples, config, device):
    source = torch.from_numpy(samples[0]['oct'])[None,None].to(device)
    target = torch.from_numpy(samples[0]['octa'])[None,None].to(device)
    records = []
    for name in ARMS:
        arm = read(ROOT/name/'config.json')
        seed_all(config['seed'])
        model = make_model(config['base_channels'], config['condition'], arm['representation']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=config['amp'] and device.type=='cuda')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type=='cuda'):
                loss, parts = objectives(model, source, target, torch.tensor([.37],device=device),
                                         arm, config['ode_steps'], config['checkpoint_rollout'])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        record = dict(arm=name, parameters=sum(p.numel() for p in model.parameters()),
                      seconds_per_attempt=(time.perf_counter()-tick)/3, losses=losses,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        records.append(record)
        print(json.dumps(record), flush=True)
        del model,optimizer,scaler,loss,parts
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(ROOT/'verification'/('benchmark_'+platform.system().lower()+'.json'),records)


def train_arm(arm_name, stage, config, selection, samples, device, resume):
    arm = read(ROOT/arm_name/'config.json')
    output = ROOT/arm_name/stage['name']
    output.mkdir(parents=True,exist_ok=True)
    evaluation, training = stage_samples(samples,stage['name'],selection)
    signature = dict(suite=config, arm=arm, stage=stage, selection_sha256=sha256(ROOT/'data/selection.json'),
                     source_sha256=source_fingerprint())
    signature_hash = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    seed_all(config['seed'])
    model = make_model(config['base_channels'],config['condition'],arm['representation']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda',init_scale=1024,enabled=config['amp'] and device.type=='cuda')
    completed, attempts, skipped, best = 0,0,0,-float('inf')
    latest = output/'latest.pt'
    if latest.exists():
        if not resume:
            raise FileExistsError(f'{latest} already exists. Use --resume.')
        saved = torch.load(latest,map_location='cpu',weights_only=False)
        if saved['signature_sha256'] != signature_hash:
            raise ValueError(f'Code, data or protocol changed: {output}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scaler.is_enabled():
            scaler.load_state_dict(saved['scaler'])
        completed,attempts,skipped,best = (saved[k] for k in ('completed_updates','attempts','skipped_updates','best_ssim'))
        del saved
        if completed >= stage['updates'] and (output/'final_metrics.json').exists():
            print(f'Already completed: {arm_name} {stage["name"]}',flush=True)
            return
    elif any(output.iterdir()):
        raise FileExistsError(f'Nonempty unfinished output without checkpoint: {output}')
    write_json(output/'protocol.json',signature)
    write_json(output/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        parameters=sum(p.numel() for p in model.parameters()),device=str(device),
        warning='Small fixed training/validation patches. Not full-volume or held-out test performance.'))
    indices,times = schedule(stage['seed'],stage['updates'],len(training))
    np.savez(output/'schedule.npz',sample_indices=indices,times=times)
    if not (output/'initial_metrics.json').exists():
        write_json(output/'initial_metrics.json',evaluate(model,evaluation,arm,config))
    model.train()
    tick = time.perf_counter()
    accumulated = []
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    while completed < stage['updates']:
        sample = training[int(indices[completed])]
        source = torch.from_numpy(sample['oct'])[None,None].to(device)
        target = torch.from_numpy(sample['octa'])[None,None].to(device)
        time_value = torch.tensor([float(times[completed])],device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=config['amp'] and device.type=='cuda'):
            loss,parts = objectives(model,source,target,time_value,arm,config['ode_steps'],config['checkpoint_rollout'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=not scaler.is_enabled())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        if scaler.get_scale() < old_scale:
            skipped += 1
            if skipped > 100:
                raise FloatingPointError('Repeated AMP overflows')
            continue  # Retry SAME example/time; every arm gets equal successful updates.
        completed += 1
        record = {key:float(value.detach()) for key,value in parts.items()}
        record.update(update=completed,attempt=attempts,case_id=sample['case_id'],patch_id=sample['patch_id'],
                      time=float(times[completed-1]))
        with (output/'steps.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        accumulated.append(record)
        if completed%25==0 or completed==stage['updates']:
            write_json(ROOT/'status.json',dict(status='running',arm=arm_name,stage=stage['name'],
                updates=completed,target_updates=stage['updates'],attempts=attempts,amp_skipped=skipped,
                seconds_this_invocation=round(time.perf_counter()-tick,2),last_loss=record['loss']))
            print(f'{arm_name} {stage["name"]} {completed}/{stage["updates"]} loss={record["loss"]:.6f}',flush=True)
        if completed%config['eval_every']==0 or completed==stage['updates']:
            report = evaluate(model,evaluation,arm,config)
            write_json(output/f'metrics_{completed:05d}.json',report)
            score = report['summary']['validation']['metrics']['ilm_opl_ssim']
            improved = score > best
            if improved:
                best = score
            payload = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                completed_updates=completed,attempts=attempts,skipped_updates=skipped,best_ssim=best,
                signature_sha256=signature_hash,protocol=signature)
            save_checkpoint(latest,payload)
            if improved:
                save_checkpoint(output/'best_validation.pt',payload)
            history = dict(update=completed,attempts=attempts,amp_skipped=skipped,
                mean_loss={key:float(np.mean([r[key] for r in accumulated])) for key in parts},
                summary=report['summary'],seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
            with (output/'history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(history,ensure_ascii=False,allow_nan=False)+'\n')
            accumulated=[]
            print(f'Validated {arm_name} {stage["name"]} update={completed} train PCC='
                  f'{report["summary"]["train"]["metrics"]["ilm_opl_pcc"]} validation SSIM={score:.4f}',flush=True)
    final = evaluate(model,evaluation,arm,config,output/'predictions')
    final.update(arm=arm_name,stage=stage['name'],completed_updates=completed,
                 attempts=attempts,amp_skipped=skipped,checkpoint='latest.pt',
                 scope='Fixed training patches and held-out validation patches; no test subjects; no whole-volume claim')
    write_json(output/'final_metrics.json',final)
    write_json(output/'status.json',dict(status='complete',completed_updates=completed,attempts=attempts,amp_skipped=skipped))
    del model,optimizer,scaler
    if device.type=='cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['all',*ARMS],default='all')
    parser.add_argument('--stage',choices=['all','one_case','few_case'],default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config,selection = read(ROOT/'suite.json'),read(ROOT/'data/selection.json')
    samples = load_samples(ROOT,selection)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.benchmark:
        benchmark(samples,config,device)
        return
    arms = list(ARMS) if args.arm=='all' else [args.arm]
    stages = [s for s in config['stages'] if args.stage in ('all',s['name'])]
    try:
        for stage in stages:
            for arm in arms:
                train_arm(arm,stage,config,selection,samples,device,args.resume)
        write_json(ROOT/'status.json',dict(status='complete',arms=arms,stages=[s['name'] for s in stages]))
    except Exception as error:
        write_json(ROOT/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
