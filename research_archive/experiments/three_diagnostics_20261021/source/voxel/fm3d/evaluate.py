"""Evaluate a fixed checkpoint on validation or test, with recorded ODE settings."""
import argparse
import json
from pathlib import Path

import torch

from .data import validate_manifest
from .model import VelocityUNet3D
from .train import evaluate, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ode-steps', type=int, help='Choose on validation, then freeze for test')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config, rows = checkpoint['config'].copy(), checkpoint['manifest']
    validate_manifest(rows, config['fov'])
    if args.ode_steps is not None:
        if args.ode_steps < 1:
            raise ValueError('ode-steps must be positive')
        config['ode_steps'] = args.ode_steps
    model = VelocityUNet3D(config['base_channels'], config['condition']).to(args.device)
    model.load_state_dict(checkpoint['model'])
    dataset = make_dataset(rows, args.split, config)
    args.output.mkdir(parents=True, exist_ok=True)
    report = evaluate(model, dataset, config, output=args.output)
    report.update({'config': config, 'checkpoint': str(args.checkpoint.resolve()),
                   'checkpoint_epoch': checkpoint['epoch'], 'split': args.split})
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
