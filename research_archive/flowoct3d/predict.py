"""OCT-only whole-volume prediction, with optional separately loaded evaluation."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fm3d.data import read_rows, validate_manifest
from fm3d.inference import generate_volume
from .data import load_case
from .metrics import aggregate, panel, sample_metrics
from .model import FlowOCT3D
from .utils import sha256, write_json


def load_source(path, axis_order, scale, depth):
    if sorted(axis_order) != ['X', 'Y', 'Z'] or not math.isfinite(scale) or scale <= 0 or depth < 0:
        raise ValueError('Invalid source preprocessing')
    with np.load(path, allow_pickle=False) as data:
        if 'axis_order' in data and str(data['axis_order'].item()) != axis_order:
            raise ValueError('Source axis metadata disagrees with configuration')
        source = np.asarray(data['oct'], dtype=np.float32)
    if source.ndim != 3 or min(source.shape) < 8 or not np.isfinite(source).all() or source.min() < 0 or source.max() > scale:
        raise ValueError('Source must be finite 3D OCT within the specified intensity scale')
    source = torch.from_numpy(source.transpose([axis_order.index(k) for k in 'ZYX']).copy())[None, None]
    if depth and source.shape[2] != depth:
        source = F.interpolate(source, size=(depth, *source.shape[-2:]), mode='trilinear', align_corners=False)
    return source/scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='NPZ containing OCT only; OCTA and labels are not required')
    inputs.add_argument('--manifest', type=Path)
    parser.add_argument('--fov', choices=('3M', '6M'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--layers-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--axis-order', default='ZYX')
    parser.add_argument('--intensity-scale', type=float, default=255.)
    parser.add_argument('--depth', type=int, default=256)
    parser.add_argument('--patch', type=int, nargs=3, default=[256, 64, 64])
    parser.add_argument('--ode-steps', type=int, default=8)
    parser.add_argument('--overlap', type=float, default=.5)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Prediction output must be empty')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = FlowOCT3D(**saved['model_config']).to(args.device).eval()
    model.load_state_dict(saved['model'])
    config = dict(axis_order=args.axis_order, intensity_scale=args.intensity_scale, depth=args.depth,
                  layers_dir=str(args.layers_dir) if args.layers_dir else None,
                  losses=saved.get('config', {}).get('losses', {'native_depth': 640}))
    # Training preprocessing is frozen when the checkpoint supplies a full-data config.
    if 'config' in saved:
        for key in ('axis_order', 'intensity_scale', 'depth', 'patch'):
            if saved['config'][key] != getattr(args, key):
                raise ValueError(f'Prediction preprocessing differs from checkpoint: {key}')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    if args.input:
        source = load_source(args.input, args.axis_order, args.intensity_scale, args.depth)
        output = generate_volume(model, source, args.patch, args.ode_steps, args.overlap,
                                 args.amp, model.representation)
        np.savez_compressed(args.output/'prediction.npz', octa=output[0, 0].numpy(), axis_order=np.array('ZYX'))
    else:
        if not args.fov:
            raise ValueError('--fov required with --manifest')
        rows = read_rows(args.manifest)
        validate_manifest(rows, args.fov)
        for row in rows:
            if row['split'] != args.split:
                continue
            # First generate from OCT alone. Load target and annotations only afterwards.
            source = load_source(row['path'], args.axis_order, args.intensity_scale, args.depth)
            output = generate_volume(model, source, args.patch, args.ode_steps, args.overlap,
                                     args.amp, model.representation)[0, 0].numpy()
            sample = load_case(row, config)
            metrics, projections = sample_metrics(output, sample, config['losses']['native_depth'], full_ssim=True)
            path = args.output/(row['case_id']+'.npz')
            np.savez_compressed(path, octa=output, octa_gt=sample['octa'], axis_order=np.array('ZYX'))
            panel(sample, output, projections, args.output/(row['case_id']+'.png'))
            records.append(dict(case_id=row['case_id'], patient_id=row['patient_id'], split=row['split'],
                                metrics=metrics, prediction_sha256=sha256(path)))
            print(f'{row["case_id"]}: MAE={metrics["volume_mae"]:.6f}', flush=True)
    write_json(args.output/'prediction.json', dict(checkpoint=str(args.checkpoint.resolve()),
               checkpoint_sha256=sha256(args.checkpoint), updates=saved['updates'], config=config,
               ode_steps=args.ode_steps, patch=args.patch, overlap=args.overlap, amp=args.amp,
               source_only_generation=True, cases=records, summary=aggregate(records) if records else None))


if __name__ == '__main__':
    main()

