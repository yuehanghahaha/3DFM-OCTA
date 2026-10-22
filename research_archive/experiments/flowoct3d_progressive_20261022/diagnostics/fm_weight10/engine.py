"""Equal-budget public FlowOCT reference and cumulative 3D removal experiments."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from flowoct.model import FlowOCT
from flowoct3d.losses import LossConfig, objectives
from flowoct3d.metrics import aggregate, panel, sample_metrics
from flowoct3d.model import FlowOCT3D
from flowoct3d.small_data import load_samples
from flowoct3d.utils import apply_update, read_json, save_checkpoint, seed_all, sha256, write_json


def to_public(volume):
    return volume.permute(0, 1, 3, 4, 2).contiguous()


def from_public(volume):
    return volume.permute(0, 1, 4, 2, 3).contiguous()


def make_public(root, device, lr=2e-4):
    options = SimpleNamespace(gpu_ids=[device.index or 0] if device.type == 'cuda' else [],
        isTrain=True, checkpoints_dir=str(root), name='public_reference', lr=lr, beta=.5,
        train_slices=0, network_chunk=0)
    return FlowOCT(options)


@torch.inference_mode()
def public_prediction(model, source, steps=8, step_clamp=False, chunk=16):
    native = to_public(source)
    b, c, d, h, w = native.shape
    flat = native.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    outputs = []
    for start in range(0, len(flat), chunk):
        state = flat[start:start+chunk]
        for i in range(steps):
            t0 = torch.full((len(state),), i/steps, device=source.device)
            t1 = torch.full_like(t0, (i+1)/steps)
            state = model.euler_step(state, t0, t1)
            if step_clamp:
                state = state.clamp(0, 1)
        outputs.append(state)
    native = torch.cat(outputs).reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)
    return from_public(native)


@torch.inference_mode()
def public_probe(model, source, target, value, chunk=16):
    c, y = [to_public(v).permute(0, 2, 1, 3, 4).reshape(-1, 1, source.shape[-1], source.shape[2])
            for v in (source, target)]
    endpoints, squared_error, elements = [], 0., 0
    for start in range(0, len(c), chunk):
        a, b = c[start:start+chunk], y[start:start+chunk]
        state = (1-value)*a+value*b
        t = torch.full((len(a),), value, device=source.device)
        coefficients = torch.cat(model.dwt_2d(state), dim=1)
        v = model.netG(coefficients, t)
        v = model.idwt_2d(v[:, :1], v[:, 1:2], v[:, 2:3], v[:, 3:4])
        squared_error += float((v-(b-a)).square().sum())
        elements += v.numel()
        endpoints.append(state+(1-value)*v)
    shape = to_public(source).shape
    endpoint = torch.cat(endpoints).reshape(shape[0], shape[2], 1, shape[3], shape[4]).permute(0, 2, 1, 3, 4)
    return squared_error/elements, from_public(endpoint)


@torch.inference_mode()
def evaluate(model, samples, arm, config, destination=None, native=False, probes=False):
    device = next(model.parameters()).device
    previous = model.training
    model.eval()
    records = []
    try:
        for sample in samples:
            source = torch.from_numpy(sample['oct'])[None, None].to(device)
            if arm['kind'] == 'public':
                raw = public_prediction(model, source, 3 if native else 8, native, config['inference_chunk'])
            else:
                with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type == 'cuda'):
                    raw = model.rollout(source, config['ode_steps'])
            if not torch.isfinite(raw).all():
                raise FloatingPointError('Nonfinite pure OCT generation')
            prediction = raw.float().clamp(0, 1)[0, 0].cpu().numpy()
            metrics, projections = sample_metrics(prediction, sample, full_ssim=destination is not None)
            metrics['raw_outside_range_fraction'] = float(((raw < 0) | (raw > 1)).float().mean())
            if probes:
                target = torch.from_numpy(sample['octa'])[None, None].to(device)
                for value, name in ((0., 't0'), (.5, 'target_assisted_t05')):
                    if arm['kind'] == 'public':
                        mse, endpoint = public_probe(model, source, target, value, config['inference_chunk'])
                    else:
                        c, y = model.encode(source), model.encode(target)
                        state = (1-value)*c+value*y
                        with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type == 'cuda'):
                            v = model(state, torch.full((1,), value, device=device), c)
                        mse = float((v.float()-(y-c)).square().mean())
                        endpoint = model.decode(state+(1-value)*v.float())
                    metrics[name+'_velocity_mse'] = mse
                    endpoint_metrics, _ = sample_metrics(endpoint.clamp(0, 1)[0, 0].cpu().numpy(), sample)
                    metrics[name+'_endpoint_pcc'] = endpoint_metrics['ilm_opl_pcc']
                    metrics[name+'_endpoint_ssim'] = endpoint_metrics['ilm_opl_ssim']
            row = {k: sample[k] for k in ('case_id', 'patient_id', 'split', 'disease', 'patch_id')}
            row['metrics'] = metrics
            if destination is not None:
                destination.mkdir(parents=True, exist_ok=True)
                name = sample['case_id']+'_'+sample['patch_id']
                path = destination/(name+'.npz')
                np.savez_compressed(path, octa=prediction, octa_gt=sample['octa'], axis_order=np.array('ZYX'))
                row['prediction_sha256'] = sha256(path)
                panel(sample, prediction, projections, destination/(name+'.png'))
            records.append(row)
    finally:
        model.train(previous)
    return dict(cases=records, summary=aggregate(records), solver='native3_per_step_clamp' if native else 'shared8_final_clamp')


def fingerprint(root, config):
    sources = {p.name: sha256(p) for p in (root/'engine.py', root/'run.py')}
    for name, digest in config['source_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    signature = dict(config=config, runner=sources)
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def train(root, name, config, samples, device, resume=False, stop_after=None):
    arm = config['arms'][name]
    output = root/name
    output.mkdir(exist_ok=True)
    training = [s for s in samples if s['split'] == 'train']
    with np.load(root/'data/schedule.npz', allow_pickle=False) as schedule:
        indices = schedule['indices'].copy()
        times = schedule['half_times' if arm.get('start_fraction') else 'times'].copy()
    signature = fingerprint(root, config)
    seed_all(config['seed'])
    if arm['kind'] == 'public':
        model = make_public(output, device, config['lr'])
        optimizer = model.optimizer_G
        scaler = torch.amp.GradScaler('cuda', enabled=False)
    else:
        model = FlowOCT3D(config['base_channels'], config['condition'], arm['representation']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=config['amp'] and device.type == 'cuda')
    completed, attempts, skipped, best, best_update = 0, 0, 0, None, None
    latest = output/'latest.pt'
    if latest.exists():
        if not resume:
            raise FileExistsError('Use --resume or a new run')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Code/data/configuration changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        completed, attempts, skipped, best, best_update = [saved[k] for k in ('updates', 'attempts', 'skipped', 'best', 'best_update')]
        for filename in ('steps.jsonl', 'history.jsonl'):
            path = output/filename
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['update'] <= completed), encoding='utf-8')
        del saved
        if completed == config['updates'] and (output/'final_metrics.json').exists():
            print('Already complete', name, flush=True)
            return
    elif any(output.iterdir()):
        raise FileExistsError('Partial run without checkpoint')
    else:
        save_checkpoint(output/'initial.pt', dict(model=model.state_dict(), updates=0))
        write_json(output/'protocol.json', dict(arm=arm, config=config, signature=signature))
    write_json(output/'environment.json', dict(torch=torch.__version__, numpy=np.__version__,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()), amp=scaler.is_enabled(),
        input_ZYX=[256, 64, 64], source_only_generation=True))
    tick = time.perf_counter()
    target_updates = min(config['updates'], stop_after) if stop_after is not None else config['updates']
    accumulated = []
    while completed < target_updates:
        sample = training[int(indices[completed])]
        source, target = [torch.from_numpy(sample[k])[None, None].to(device) for k in ('oct', 'octa')]
        if arm['kind'] == 'public':
            update_seed = 20261023*10000+completed+1
            seed_all(update_seed)
            model.set_input(dict(A=to_public(source), B=to_public(target)))
            model.forward()
            parts = {k: float(v) for k, v in model.get_current_loss().items()}
            parts['loss'] = float(model.loss.detach())
            expected = 10*parts['vloss']+.5*parts['L1loss']+.005*parts['contrastloss']+.1*parts['proj_loss']
            if abs(expected-parts['loss']) > 1e-6*max(1., abs(expected)):
                raise AssertionError('Public loss weights changed')
            extra = dict(seed=update_seed, sampled_Bscans=source.shape[3]//4)
            overflow = False
        else:
            optimizer.zero_grad(set_to_none=True)
            t = torch.tensor([float(times[completed])], device=device)
            layers = torch.from_numpy(sample['layers'])[None].to(device)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, tensors = objectives(model, source, target, t, LossConfig(**arm['losses']), layers)
            overflow, grad_norm = apply_update(loss, model, optimizer, scaler)
            parts = {k: float(v.detach()) for k, v in tensors.items()}
            extra = dict(time=float(times[completed]), grad_norm=grad_norm)
        attempts += 1
        if overflow:
            skipped += 1
            if skipped > 100:
                raise FloatingPointError('Repeated AMP overflow')
            continue
        completed += 1
        record = dict(update=completed, attempt=attempts, case_id=sample['case_id'], patch_id=sample['patch_id'], **parts, **extra)
        with (output/'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        accumulated.append(parts)
        if completed % 50 == 0 or completed == target_updates:
            write_json(root/'status.json', dict(status='running', arm=name, updates=completed,
                target=config['updates'], attempts=attempts, skipped=skipped, seconds=time.perf_counter()-tick, loss=parts['loss']))
            print(name, completed, '/', config['updates'], 'loss', parts['loss'], flush=True)
        if completed % config['eval_every'] == 0 or completed == target_updates:
            milestone = completed in config['milestones']
            destination = output/f'predictions_{completed:05d}' if milestone else None
            result = evaluate(model, samples, arm, config, destination, probes=completed == config['updates'])
            result.update(arm=name, updates=completed, scope=config['scope'])
            score = result['summary']['validation']['metrics']['ilm_opl_ssim']
            improved = best is None or score > best
            if improved:
                best, best_update = score, completed
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                updates=completed, attempts=attempts, skipped=skipped, best=best, best_update=best_update,
                signature=signature, config=config, arm=name)
            save_checkpoint(latest, payload)
            if milestone:
                save_checkpoint(output/f'update_{completed:05d}.pt', payload)
            if improved:
                save_checkpoint(output/'best_validation.pt', payload)
            if completed == config['updates']:
                result.update(checkpoint_sha256=sha256(latest), optimizer_steps=sorted({int(s['step']) for s in optimizer.state.values()}),
                              attempts=attempts, skipped=skipped)
                write_json(output/'final_metrics.json', result)
            write_json(output/f'metrics_{completed:05d}.json', result)
            history = dict(update=completed, summary=result['summary'], seconds=time.perf_counter()-tick,
                           losses={k: float(np.mean([r[k] for r in accumulated])) for k in parts})
            with (output/'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(history, allow_nan=False)+'\n')
            accumulated = []
            print('Validated', name, completed, 'train PCC', result['summary']['train']['metrics']['ilm_opl_pcc'],
                  'validation PCC/SSIM', result['summary']['validation']['metrics']['ilm_opl_pcc'], score, flush=True)
    if completed == config['updates']:
        if arm['kind'] == 'public':
            native = evaluate(model, samples, arm, config, output/'predictions_native3', native=True)
            native.update(arm=name, updates=completed)
            write_json(output/'final_native3.json', native)
        write_json(output/'status.json', dict(status='complete', updates=completed, attempts=attempts, skipped=skipped))
    del model, optimizer, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()

