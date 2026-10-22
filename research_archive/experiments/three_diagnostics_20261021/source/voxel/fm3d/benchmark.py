"""Measure synthetic forward/backward/Adam memory on the actual training GPU."""
import argparse
import json
import time
from pathlib import Path

import torch

from .model import VelocityUNet3D, flow_loss
from .train import apply_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch', type=int, nargs=3, default=[256, 64, 64])
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.patch) < 8 or min(args.steps, args.batch_size) < 1:
        raise ValueError('patch>=8, steps>=1, batch_size>=1 required')
    device = torch.device(args.device)
    torch.manual_seed(2026)
    model = VelocityUNet3D(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=args.amp and device.type == 'cuda')
    source = torch.rand(args.batch_size, 1, *args.patch, device=device)
    target = torch.rand_like(source)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    timings, losses, skipped_updates = [], [], 0
    for step in range(args.steps + 1):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
            loss = flow_loss(model, source, target)
        skipped_updates += int(apply_update(loss, model, optimizer, scaler))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if step:
            timings.append(time.perf_counter() - tick)
        losses.append(loss.item())
    report = {'scope': 'Synthetic GPU/CPU resource check only; not patient performance',
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'torch': torch.__version__, 'patch_ZYX': args.patch, 'batch_size': args.batch_size,
              'base_channels': args.base_channels, 'amp': args.amp and device.type == 'cuda',
              'parameters': sum(p.numel() for p in model.parameters()),
              'optimizer_updates_including_warmup': args.steps + 1 - skipped_updates,
              'amp_skipped_updates': skipped_updates,
              'mean_update_seconds_after_warmup': sum(timings) / len(timings),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'losses': losses}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
