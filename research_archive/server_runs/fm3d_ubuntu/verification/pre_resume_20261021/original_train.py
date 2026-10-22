"""Train a paired 3D voxel flow; full-volume validation, checkpointing and resume."""
import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .data import COUNTS, VolumeDataset, read_rows, validate_manifest
from .inference import generate_volume
from .model import VelocityUNet3D, flow_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def apply_update(loss, model, optimizer, scaler):
    """Apply one update; return True when AMP overflow skipped the optimizer step."""
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # AMP recorded overflow during unscale; allow its automatic skip/scale reduction.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() < old_scale


def make_dataset(rows, phase, config, training=False):
    return VolumeDataset(rows, phase, config['axis_order'], config['intensity_scale'],
                         config['depth'], config['patch'] if training else None)


@torch.inference_mode()
def evaluate(model, dataset, config, output=None):
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        source, target = sample['oct'][None], sample['octa'][None]
        prediction = generate_volume(model, source, config['patch'], config['ode_steps'],
                                     config['overlap'], config['amp'])
        err = prediction - target
        mse = err.square().mean(dtype=torch.float64).item()
        record = {'case_id': sample['case_id'], 'patient_id': dataset.rows[index]['patient_id'],
                  'mae': err.abs().mean(dtype=torch.float64).item(),
                  'psnr': -10 * math.log10(max(mse, 1e-12))}
        records.append(record)
        if output is not None:
            np.savez_compressed(output / f'{sample["case_id"]}.npz',
                                octa=prediction[0, 0].numpy(), octa_gt=target[0, 0].numpy(),
                                axis_order=np.array('ZYX'))
        print(f'{index + 1}/{len(dataset)} {sample["case_id"]}: '
              f'MAE={record["mae"]:.6f} PSNR={record["psnr"]:.3f}', flush=True)
    patients = defaultdict(list)
    for row in records:
        patients[row['patient_id']].append(row)
    metrics = {key: float(np.mean([np.mean([r[key] for r in scans])
                                   for scans in patients.values()])) for key in ('mae', 'psnr')}
    return {'metrics': metrics, 'cases': records, 'patients': len(patients),
            'protocol': 'full preprocessed volume; range=1; final clamp; patient macro mean; PSNR MSE floor=1e-12'}


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fov', choices=COUNTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axis-order', required=True, help='NPZ axes, e.g. ZYX or YXZ; never inferred')
    parser.add_argument('--intensity-scale', type=float, required=True, help='255 for native uint8, 1 for verified [0,1]')
    parser.add_argument('--depth', type=int, default=256, help='Resample axial Z only; 0 preserves native depth')
    parser.add_argument('--patch', type=int, nargs=3, default=[256, 64, 64], metavar=('Z', 'Y', 'X'))
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--condition', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=200, help='Training batches; random cases with replacement; AMP-skipped updates logged')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--ode-steps', type=int, default=8)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true', help='Resume local latest.pt at next epoch')
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.patch) < 8 or args.depth < 0 or (args.depth and args.patch[0] > args.depth):
        raise ValueError('Invalid patch/depth: patch>=8; depth=0 or depth>=patch Z')
    if min(args.epochs, args.steps_per_epoch, args.val_every, args.ode_steps, args.batch_size) < 1:
        raise ValueError('Epoch/step/frequency/batch counts must be positive')
    if not 0 <= args.overlap < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid overlap, workers or learning rate')
    if sorted(args.axis_order.upper()) != ['X', 'Y', 'Z'] or not math.isfinite(args.intensity_scale) or args.intensity_scale <= 0:
        raise ValueError('Explicit valid axis_order and finite positive intensity_scale required')
    device = torch.device(args.device)
    rows = read_rows(args.manifest)
    validate_manifest(rows, args.fov)
    config = vars(args).copy()
    config['axis_order'] = args.axis_order.upper()
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in config.items()}
    config['manifest_sha256'] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    seed_all(args.seed)
    model = VelocityUNet3D(args.base_channels, args.condition).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=args.amp and device.type == 'cuda')
    start_epoch, best_mae = 1, float('inf')
    latest = args.output / 'latest.pt'
    if args.resume:
        # Load only checkpoints created by this run (contains optimizer/config, not just weights).
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        mutable = {'epochs', 'workers', 'device', 'resume'}
        for key, value in config.items():
            if key not in mutable and checkpoint['config'].get(key) != value:
                raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler.is_enabled() and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best_mae = checkpoint['epoch'] + 1, checkpoint['best_mae']
        del checkpoint
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output is nonempty; use a new run directory or --resume')
    if start_epoch > args.epochs:
        raise ValueError('Requested epochs already completed; increase --epochs to resume')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (args.output / 'manifest_snapshot.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (args.output / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'parameters': sum(p.numel() for p in model.parameters()),
        'reproducibility': 'Epoch-specific RNG seeds; bitwise determinism across GPU/software not guaranteed'
    }, indent=2), encoding='utf-8')
    training = make_dataset(rows, 'train', config, training=True)
    validation = make_dataset(rows, 'validation', config)
    for epoch in range(start_epoch, args.epochs + 1):
        seed_all(args.seed + epoch)
        sampler = RandomSampler(training, replacement=True, num_samples=args.steps_per_epoch * args.batch_size)
        loader = DataLoader(training, sampler=sampler, batch_size=args.batch_size,
                            num_workers=args.workers, worker_init_fn=seed_worker,
                            pin_memory=device.type == 'cuda')
        model.train()
        losses, tick, skipped_updates = [], time.perf_counter(), 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for step, sample in enumerate(loader, 1):
            source, target = sample['oct'].to(device), sample['octa'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                loss = flow_loss(model, source, target)
            skipped_updates += int(apply_update(loss, model, optimizer, scaler))
            losses.append(loss.item())
            if step == 1 or step % 20 == 0 or step == args.steps_per_epoch:
                print(f'epoch={epoch} step={step}/{args.steps_per_epoch} FM={loss.item():.6f}', flush=True)
        if skipped_updates == args.steps_per_epoch:
            raise FloatingPointError('All AMP updates skipped; inspect gradients or retry with --no-amp')
        record = {'epoch': epoch, 'fm_loss': float(np.mean(losses)), 'train_seconds': time.perf_counter() - tick,
                  'optimizer_updates': args.steps_per_epoch - skipped_updates,
                  'amp_skipped_updates': skipped_updates, 'amp_scale': scaler.get_scale(),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
        improved = False
        if epoch % args.val_every == 0 or epoch == args.epochs:
            report = evaluate(model, validation, config)
            (args.output / f'validation_{epoch:04d}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            record['validation'] = report['metrics']
            improved = report['metrics']['mae'] < best_mae
            best_mae = min(best_mae, report['metrics']['mae'])
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                   'epoch': epoch, 'best_mae': best_mae, 'config': config, 'manifest': rows}
        save_checkpoint(latest, payload)
        if improved:
            save_checkpoint(args.output / 'best.pt', payload)
        with (args.output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
