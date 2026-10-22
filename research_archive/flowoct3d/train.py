"""Whole-dataset training with 3D FlowOCT losses and whole-volume validation."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from fm3d.data import read_rows, validate_manifest
from fm3d.inference import generate_volume
from fm3d.wavelets import validate_wavelet_patch
from .data import crop_case, load_case
from .losses import LossConfig, objectives
from .metrics import aggregate, sample_metrics
from .model import FlowOCT3D
from .utils import apply_update, code_hashes, read_json, save_checkpoint, seed_all, sha256, write_json


@torch.inference_mode()
def validate(model, rows, config):
    records = []
    for row in rows:
        sample = load_case(row, config)
        source = torch.from_numpy(sample['oct'])[None, None]
        prediction = generate_volume(model, source, config['patch'], config['ode_steps'],
                                     config['overlap'], config['amp'], config['representation'])
        metrics, _ = sample_metrics(prediction[0, 0].numpy(), sample, config['losses']['native_depth'])
        records.append(dict(case_id=row['case_id'], patient_id=row['patient_id'], split=row['split'], metrics=metrics))
        print(f'Validation {row["case_id"]}: MAE={metrics["volume_mae"]:.6f}', flush=True)
    return dict(cases=records, summary=aggregate(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fov', choices=('3M', '6M'), required=True)
    parser.add_argument('--layers-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loss-config', type=Path)
    parser.add_argument('--representation', choices=('voxel', 'haar3d'), default='haar3d')
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--condition', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--axis-order', default='ZYX')
    parser.add_argument('--intensity-scale', type=float, default=255.)
    parser.add_argument('--depth', type=int, default=256)
    parser.add_argument('--patch', type=int, nargs=3, default=[256, 64, 64])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps-per-epoch', type=int, default=200)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--ode-steps', type=int, default=8)
    parser.add_argument('--overlap', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--start-fraction', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cache-cases', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    loss_config = LossConfig(**read_json(args.loss_config)) if args.loss_config else LossConfig()
    if min(args.epochs, args.steps_per_epoch, args.val_every, args.ode_steps, args.cache_cases) < 1:
        raise ValueError('Epochs, steps, frequencies and cache size must be positive')
    if not 0 <= args.start_fraction <= 1 or not 0 <= args.overlap < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid start fraction, overlap or learning rate')
    if args.depth < 16 or args.patch[0] != args.depth or min(args.patch) < 16:
        raise ValueError('patch must span depth; all dimensions>=16')
    if args.representation == 'haar3d':
        validate_wavelet_patch(args.patch)
    if loss_config.projection_weight and loss_config.projection == 'retinal' and not args.layers_dir:
        raise ValueError('--layers-dir required for anatomical projection training')
    rows = read_rows(args.manifest)
    validate_manifest(rows, args.fov)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['losses'] = loss_config.to_dict()
    config['manifest_sha256'] = sha256(args.manifest)
    config['source_hashes'] = code_hashes()
    config['batch_size'] = 1
    # Audit native annotation files actually used for training or validation.
    config['layer_hashes'] = {r['case_id']: sha256(args.layers_dir/(r['case_id']+'.mat'))
                              for r in rows if r['split'] in ('train', 'validation')} if args.layers_dir else {}
    device = torch.device(args.device)
    seed_all(args.seed)
    model = FlowOCT3D(args.base_channels, args.condition, args.representation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=args.amp and device.type == 'cuda')
    first_epoch, updates, attempts, best = 1, 0, 0, None
    latest = args.output/'latest.pt'
    mutable = {'epochs', 'resume', 'device', 'output', 'cache_cases', 'loss_config'}
    if args.resume:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        for key, value in config.items():
            if key not in mutable and saved['config'].get(key) != value:
                raise ValueError(f'Resume protocol changed: {key}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        first_epoch, updates, attempts, best = saved['epoch']+1, saved['updates'], saved['attempts'], saved['best']
        for filename in ('steps.jsonl', 'epochs.jsonl'):
            path = args.output/filename
            if path.exists():
                committed = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                committed = [r for r in committed if r['epoch'] < first_epoch]
                path.write_text(''.join(json.dumps(r)+'\n' for r in committed), encoding='utf-8')
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output is not empty; use --resume or a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if first_epoch > args.epochs:
        raise ValueError('Training budget already completed; increase --epochs to resume')
    write_json(args.output/'config.json', config)
    write_json(args.output/'manifest_snapshot.json', rows)
    write_json(args.output/'environment.json', dict(torch=torch.__version__, python=platform.python_version(),
               device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None))
    training, validation = [[r for r in rows if r['split'] == phase] for phase in ('train', 'validation')]
    cache = OrderedDict()
    for epoch in range(first_epoch, args.epochs+1):
        seed_all(args.seed+epoch)
        rng = np.random.default_rng(args.seed+epoch)
        time_rng = np.random.default_rng(args.seed+epoch+100000)
        times = time_rng.random(args.steps_per_epoch).astype(np.float32)
        start_ids = time_rng.permutation(args.steps_per_epoch)[:round(args.steps_per_epoch*args.start_fraction)]
        times[start_ids] = 0.
        losses, skipped = [], 0
        tick = time.perf_counter()
        model.train()
        for step in range(args.steps_per_epoch):
            row = training[int(rng.integers(len(training)))]
            cid = row['case_id']
            if cid not in cache:
                cache[cid] = load_case(row, config, with_layers=bool(loss_config.projection_weight))
                if len(cache) > args.cache_cases:
                    cache.popitem(last=False)
            cache.move_to_end(cid)
            sample = crop_case(cache[cid], args.patch, rng, args.representation)
            source, target = [torch.from_numpy(sample[k])[None, None].to(device) for k in ('oct', 'octa')]
            layers = torch.from_numpy(sample['layers'])[None].to(device) if 'layers' in sample else None
            t = torch.tensor([float(times[step])], device=device)
            while True:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss, parts = objectives(model, source, target, t, loss_config, layers)
                overflow, grad_norm = apply_update(loss, model, optimizer, scaler)
                attempts += 1
                if not overflow:
                    break
                skipped += 1
                if skipped > 100:
                    raise FloatingPointError('Repeated AMP overflows')
            updates += 1
            record = {k: float(v.detach()) for k, v in parts.items()}
            record.update(epoch=epoch, step=step+1, update=updates, attempt=attempts,
                          case_id=cid, origin_ZYX=sample['origin_ZYX'], time=float(t), grad_norm=grad_norm)
            with (args.output/'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            losses.append(record['loss'])
        result, improved = None, False
        if epoch % args.val_every == 0 or epoch == args.epochs:
            result = validate(model, validation, config)
            metrics = result['summary']['validation']['metrics']
            score = metrics['ilm_opl_ssim'] if args.layers_dir else -metrics['volume_mae']
            improved = best is None or score > best
            best = score if improved else best
            write_json(args.output/f'validation_{epoch:04d}.json', result)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                       config=config, model_config=dict(base_channels=args.base_channels, condition=args.condition,
                                                       representation=args.representation),
                       epoch=epoch, updates=updates, attempts=attempts, best=best)
        save_checkpoint(latest, payload)
        if improved:
            save_checkpoint(args.output/'best.pt', payload)
        history = dict(epoch=epoch, updates=updates, attempts=attempts, amp_skipped=skipped,
                       mean_loss=float(np.mean(losses)), seconds=time.perf_counter()-tick,
                       validation=result['summary'] if result else None)
        with (args.output/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(history, allow_nan=False)+'\n')
        print(f'Epoch {epoch}/{args.epochs} updates={updates} loss={history["mean_loss"]:.6f}', flush=True)


if __name__ == '__main__':
    main()

