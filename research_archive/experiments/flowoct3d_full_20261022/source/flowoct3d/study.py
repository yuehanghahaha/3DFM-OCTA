"""Reproducible small-patient suite with actual-update budgets and ablations."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch

from .losses import LossConfig, objectives
from .metrics import aggregate, panel, sample_metrics
from .model import FlowOCT3D
from .small_data import load_samples
from .utils import (apply_update, code_hashes, read_json, save_checkpoint,
                    seed_all, sha256, write_json)


def arms():
    full = LossConfig()
    bare = replace(full, reconstruction_weight=0., projection_weight=0., rdca_weight=0., structure=False)
    return {
        '01_voxel_fm': dict(representation='voxel', losses=bare.to_dict(), start_fraction=0.),
        '02_haar_fm': dict(representation='haar3d', losses=bare.to_dict(), start_fraction=0.),
        '03_full': dict(representation='haar3d', losses=full.to_dict(), start_fraction=0.),
        '04_no_rdca': dict(representation='haar3d', losses=replace(full, rdca_weight=0.).to_dict(), start_fraction=0.),
        '05_no_projection': dict(representation='haar3d', losses=replace(full, projection_weight=0.).to_dict(), start_fraction=0.),
        '06_no_structure': dict(representation='haar3d', losses=replace(full, structure=False).to_dict(), start_fraction=0.),
        '07_no_reconstruction': dict(representation='haar3d', losses=replace(full, reconstruction_weight=0.).to_dict(), start_fraction=0.),
        '08_full_half_start': dict(representation='haar3d', losses=full.to_dict(), start_fraction=.5),
    }


def default_config():
    return dict(schema='flowoct3d_full_v1', seed=20261021, base_channels=16, condition=True,
                lr=2e-4, amp=True, ode_steps=8, eval_every=200, patch_ZYX=[256, 64, 64],
                stages=[dict(name='one_case', updates=400, seed=20261022,
                             arms=['01_voxel_fm', '02_haar_fm', '03_full', '08_full_half_start']),
                        dict(name='few_case', updates=800, seed=20261023, arms=list(arms()))],
                arms=arms(), selection='Validation patient-macro ILM-OPL SSIM; primary comparison uses final equal-update weights',
                scope='One seed; fixed full-Z patches; no test set, no whole-volume or clinical efficacy claim')


def schedule(seed, updates, count, start_fraction=0.):
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.permutation(count) for _ in range(math.ceil(updates/count))])[:updates]
    times = rng.random(updates).astype(np.float32)
    if start_fraction:
        mask_rng = np.random.default_rng(seed+991)
        selected = mask_rng.permutation(updates)[:round(updates*start_fraction)]
        times[selected] = 0.
    return indices, times


def stage_samples(samples, selection, name):
    if name == 'few_case':
        return samples, [s for s in samples if s['split'] == 'train']
    training = [s for s in samples if s['case_id']+'_'+s['patch_id'] == selection['one_case_patch']]
    if len(training) != 1:
        raise ValueError('One-case training patch is missing or duplicated')
    return training+[s for s in samples if s['split'] == 'validation'], training


@torch.inference_mode()
def evaluate(model, samples, config, destination=None):
    device = next(model.parameters()).device
    previous = model.training
    model.eval()
    records = []
    try:
        for sample in samples:
            source = torch.from_numpy(sample['oct'])[None, None].to(device)
            target = torch.from_numpy(sample['octa'])[None, None].to(device)
            with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type == 'cuda'):
                raw = model.rollout(source, config['ode_steps'])
            if not torch.isfinite(raw).all():
                raise FloatingPointError('Nonfinite OCT-only generated volume')
            prediction = raw.float().clamp(0, 1)[0, 0].cpu().numpy()
            metrics, projections = sample_metrics(prediction, sample, full_ssim=destination is not None)
            metrics['raw_outside_range_fraction'] = float(((raw < 0) | (raw > 1)).float().mean())
            condition, truth = model.encode(source), model.encode(target)
            for value, name in ((0., 't0_velocity_mse'), (.5, 'target_assisted_t05_velocity_mse')):
                t = torch.full((1,), value, device=device)
                with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type == 'cuda'):
                    v = model((1-value)*condition+value*truth, t, condition)
                metrics[name] = float((v.float()-(truth-condition)).square().mean())
            record = {k: sample[k] for k in ('case_id', 'patient_id', 'split', 'disease', 'patch_id')}
            record['metrics'] = metrics
            if destination is not None:
                destination.mkdir(parents=True, exist_ok=True)
                name = sample['case_id']+'_'+sample['patch_id']
                path = destination/(name+'.npz')
                np.savez_compressed(path, octa=prediction, octa_gt=sample['octa'], axis_order=np.array('ZYX'))
                record['prediction_sha256'] = sha256(path)
                panel(sample, prediction, projections, destination/(name+'.png'))
            records.append(record)
    finally:
        model.train(previous)
    return dict(cases=records, summary=aggregate(records))


def train_arm(root, name, stage, config, selection, samples, device, resume=False, stop_after=None):
    root = Path(root)
    arm = config['arms'][name]
    loss_config = LossConfig(**arm['losses'])
    evaluation, training = stage_samples(samples, selection, stage['name'])
    output = root/name/stage['name']
    output.mkdir(parents=True, exist_ok=True)
    indices, times = schedule(stage['seed'], stage['updates'], len(training), arm['start_fraction'])
    signature = dict(config=config, stage=stage, arm=name, sources=code_hashes(),
                     selection_sha256=sha256(root/'data/selection.json'),
                     schedule_sha256=hashlib.sha256(indices.tobytes()+times.tobytes()).hexdigest())
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    seed_all(config['seed'])
    model = FlowOCT3D(config['base_channels'], config['condition'], arm['representation']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=config['amp'] and device.type == 'cuda')
    completed, attempts, skipped, best, best_update = 0, 0, 0, None, None
    latest = output/'latest.pt'
    if latest.exists():
        if not resume:
            raise FileExistsError(f'{latest}: use --resume')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['signature_sha256'] != fingerprint:
            raise ValueError('Code, data or training configuration changed; use a new run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        completed, attempts, skipped = saved['updates'], saved['attempts'], saved['skipped']
        best, best_update = saved['best'], saved['best_update']
        # Remove uncommitted log suffix after a crash, only within this run's logs.
        for filename in ('steps.jsonl', 'history.jsonl'):
            path = output/filename
            if path.exists():
                rows = [read for line in path.read_text().splitlines() if line.strip()
                        for read in [json.loads(line)] if read['update'] <= completed]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        del saved
    elif any(p.name not in ('protocol.json',) for p in output.iterdir()):
        raise FileExistsError('Incomplete run without checkpoint; choose another output directory')
    else:
        write_json(output/'protocol.json', signature)
        np.savez_compressed(output/'schedule.npz', indices=indices, times=times)
        save_checkpoint(output/'initial.pt', dict(model=model.state_dict(), updates=0))
        initial = evaluate(model, evaluation, config)
        write_json(output/'initial_metrics.json', initial)
    environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                       device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                       velocity_parameters=sum(p.numel() for p in model.velocity.parameters()),
                       rdca_parameters=sum(p.numel() for p in model.depth_encoder.parameters()),
                       amp=scaler.is_enabled(), source_only_inference=True)
    write_json(output/'environment.json', environment)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    accumulated = []
    target_updates = min(stage['updates'], stop_after) if stop_after is not None else stage['updates']
    while completed < target_updates:
        sample = training[int(indices[completed])]
        source = torch.from_numpy(sample['oct'])[None, None].to(device)
        target = torch.from_numpy(sample['octa'])[None, None].to(device)
        layers = torch.from_numpy(sample['layers'])[None].to(device)
        t = torch.tensor([float(times[completed])], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss, parts = objectives(model, source, target, t, loss_config, layers)
        overflow, grad_norm = apply_update(loss, model, optimizer, scaler)
        attempts += 1
        if overflow:
            skipped += 1
            if skipped > 100:
                raise FloatingPointError('Repeated AMP overflow')
            continue
        completed += 1
        record = {key: float(value.detach()) for key, value in parts.items()}
        record.update(update=completed, attempt=attempts, grad_norm=grad_norm, case_id=sample['case_id'],
                      patch_id=sample['patch_id'], time=float(times[completed-1]))
        with (output/'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        accumulated.append(record)
        if completed % 25 == 0 or completed == target_updates:
            write_json(root/'status.json', dict(status='running', stage=stage['name'], arm=name,
                       updates=completed, target=stage['updates'], attempts=attempts, skipped=skipped,
                       seconds=time.perf_counter()-tick, loss=record['loss']))
            print(f'{stage["name"]} {name}: {completed}/{stage["updates"]} loss={record["loss"]:.6f}', flush=True)
        if completed % config['eval_every'] == 0 or completed == target_updates:
            report = evaluate(model, evaluation, config)
            write_json(output/f'metrics_{completed:05d}.json', report)
            score = report['summary']['validation']['metrics']['ilm_opl_ssim']
            improved = best is None or score > best
            if improved:
                best, best_update = score, completed
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                           updates=completed, attempts=attempts, skipped=skipped, best=best,
                           best_update=best_update, signature_sha256=fingerprint, protocol=signature,
                           model_config=dict(base_channels=config['base_channels'], condition=config['condition'],
                                             representation=arm['representation']))
            save_checkpoint(latest, payload)
            if improved:
                save_checkpoint(output/'best_validation.pt', payload)
            history = dict(update=completed, attempts=attempts, amp_skipped=skipped, summary=report['summary'],
                           losses={k: float(np.mean([r[k] for r in accumulated])) for k in parts},
                           seconds=time.perf_counter()-tick,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
            with (output/'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(history, allow_nan=False)+'\n')
            accumulated = []
            print(f'Validation ILM-OPL SSIM={score:.4f}; train PCC='
                  f'{report["summary"]["train"]["metrics"]["ilm_opl_pcc"]:.4f}', flush=True)
    if completed == stage['updates']:
        final = evaluate(model, evaluation, config, output/'predictions')
        optimizer_steps = sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})
        if optimizer_steps != [completed]:
            raise AssertionError(f'Optimizer step counts differ from successful-update budget: {optimizer_steps}')
        final.update(arm=name, stage=stage['name'], updates=completed, attempts=attempts, skipped=skipped,
                     checkpoint='latest.pt', checkpoint_sha256=sha256(latest),
                     best_validation_update=best_update, optimizer_steps=optimizer_steps,
                     scope=config['scope'])
        write_json(output/'final_metrics.json', final)
        write_json(output/'status.json', dict(status='complete', updates=completed))
    else:
        write_json(output/'status.json', dict(status='checkpointed', updates=completed))
    del model, optimizer, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def benchmark(root, config, samples, device):
    sample = samples[0]
    source, target = [torch.from_numpy(sample[k])[None, None].to(device) for k in ('oct', 'octa')]
    layers = torch.from_numpy(sample['layers'])[None].to(device)
    records = []
    for name in ('01_voxel_fm', '02_haar_fm', '03_full'):
        arm = config['arms'][name]
        seed_all(config['seed'])
        model = FlowOCT3D(config['base_channels'], config['condition'], arm['representation']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=config['amp'] and device.type == 'cuda')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tick, updates, attempts = time.perf_counter(), 0, 0
        while updates < 3:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, parts = objectives(model, source, target, torch.tensor([.37], device=device), LossConfig(**arm['losses']), layers)
            skipped, _ = apply_update(loss, model, optimizer, scaler)
            attempts += 1
            updates += int(not skipped)
            if attempts > 20:
                raise FloatingPointError('Repeated benchmark AMP overflows')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        record = dict(arm=name, updates=updates, attempts=attempts,
                      seconds_per_update=(time.perf_counter()-tick)/updates,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                      losses={k: float(v.detach()) for k, v in parts.items()})
        records.append(record)
        print(json.dumps(record), flush=True)
        del model, optimizer, scaler, loss, parts
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(root/'verification/benchmark.json', records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('all', 'one_case', 'few_case'), default='all')
    parser.add_argument('--arm', choices=('all', *arms()), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = args.root/'suite.json'
    if not config_path.exists():
        write_json(config_path, default_config())
    config = read_json(config_path)
    selection, samples = load_samples(args.root)
    device = torch.device(args.device)
    if args.benchmark:
        benchmark(args.root, config, samples, device)
        return
    try:
        for stage in config['stages']:
            if args.stage not in ('all', stage['name']):
                continue
            names = stage['arms'] if args.arm == 'all' else [args.arm]
            for name in names:
                train_arm(args.root, name, stage, config, selection, samples, device, args.resume, args.stop_after)
        write_json(args.root/'status.json', dict(status='checkpointed' if args.stop_after else 'complete',
                   stage=args.stage, arm=args.arm))
    except Exception as error:
        write_json(args.root/'status.json', dict(status='failed', error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

