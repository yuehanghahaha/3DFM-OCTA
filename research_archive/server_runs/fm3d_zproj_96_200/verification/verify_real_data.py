"""Preflight on a training case only; no validation/test data used for updates."""
import json
import time
from pathlib import Path

import torch

from fm3d.data import VolumeDataset, read_rows, validate_manifest
from fm3d.inference import generate_volume
from fm3d.model import flow_loss, make_model, z_mean_projection, z_projection_loss
from fm3d.train import apply_update, seed_all


def main():
    torch.set_num_threads(4)
    seed_all(2026)
    manifest = Path('/docsystem/niuke/lvyuehang/OCTA500_common_v1/split_3M_z256.csv')
    rows = read_rows(manifest)
    validate_manifest(rows, '3M')
    dataset = VolumeDataset(rows, 'train', 'ZYX', 255, 256, (256, 96, 96))
    sample = dataset[0]
    source, target = sample['oct'][None].cuda(), sample['octa'][None].cuda()
    model = make_model(16).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    torch.cuda.reset_peak_memory_stats()
    steps = []
    tick = time.perf_counter()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            fm = flow_loss(model, source, target)
            projection = z_projection_loss(model, source, target)
            loss = fm + 0.1 * projection
        skipped = apply_update(loss, model, optimizer, scaler)
        steps.append(dict(step=step + 1, fm_loss=fm.item(), z_projection_loss=projection.item(),
                          loss=loss.item(), amp_skipped_update=skipped))
    prediction = generate_volume(model, source.cpu(), (256, 96, 96), steps=8, amp=True)
    reference = target.cpu()
    assert torch.isfinite(prediction).all()
    assert any(not step['amp_skipped_update'] for step in steps)
    report = dict(scope='Three updates on one training patch and eight-step patch inference; not performance validation',
                  case_id=sample['case_id'], split='train', gpu=torch.cuda.get_device_name(),
                  patch_ZYX=list(source.shape[2:]), source_range=[source.min().item(), source.max().item()],
                  target_range=[target.min().item(), target.max().item()], steps=steps,
                  prediction_shape=list(prediction.shape), mae=(prediction - reference).abs().mean().item(),
                  z_projection_mae=(z_mean_projection(prediction) - z_mean_projection(reference)).abs().mean().item(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.perf_counter() - tick)
    output = Path('verification/real_data_smoke.json')
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
