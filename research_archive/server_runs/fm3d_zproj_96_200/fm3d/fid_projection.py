"""Exploratory clean-FID on one raw ILM-OPL projection per test patient."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W
from .metrics_suite import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    files = sorted(args.input.glob('*_projections.npz'))
    if len(files) != 50:
        raise ValueError('Expected 50 cases')
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            for key in ['prediction', 'reference_z256']:
                folder = args.output / key
                folder.mkdir(parents=True, exist_ok=True)
                arr = np.rint(data[key + '_ILM_OPL'] * 255).astype(np.uint8)
                Image.fromarray(arr).convert('RGB').save(folder / (file.stem.split('_')[0] + '.png'))
    model = InceptionV3W(str(args.weights), download=False).eval().to('cuda:0')
    features = {}
    for key in ['prediction', 'reference_z256']:
        # Explicit list avoids clean-fid globbing both *.png and *.PNG on Windows.
        image_files = [str(args.output / key / (f.stem.split('_')[0] + '.png')) for f in files]
        features[key] = fid.get_files_features(image_files, model=model,
            num_workers=0, batch_size=8, device=torch.device('cuda:0'), mode='clean')
    np.savez_compressed(args.output / 'features.npz', **features,
                        case_ids=np.array([f.stem.split('_')[0] for f in files]))
    a, b = [features[k].astype(np.float64) for k in ['prediction', 'reference_z256']]
    score = fid.frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
    # Independent low-rank identity: trace(sqrt(Ca Cb)) = nuclear_norm(A B^T)/(n-1).
    ac, bc = a - a.mean(0), b - b.mean(0)
    lowrank = float(np.sum((a.mean(0) - b.mean(0)) ** 2) + np.sum(ac**2)/49 + np.sum(bc**2)/49
                    - 2 * np.linalg.svd(ac @ bc.T / 49, compute_uv=False).sum())
    if not np.isclose(score, lowrank, rtol=1e-5, atol=1e-4):
        raise AssertionError('FID covariance and low-rank calculation disagree')
    result = dict(fid=float(score), independent_lowrank_check=lowrank,
        n_prediction=len(a), n_reference=len(b), feature_dimension=2048,
        unit='One ILM_OPL layer maximum projection per patient,50 patients; no Bscan pooling',
        preprocessing='Raw[0,1] -> round(255*x)uint8 -> replicatedRGB PNG -> clean-fid PIL bicubic299x299; no contrast enhancement',
        network='NVIDIA/StyleGAN2-ADA InceptionV3; clean-fid clean mode',
        weights_sha256=sha(args.weights / 'inception-2015-12-05.pt'),
        clean_fid_version=importlib.metadata.version('clean-fid'),
        limitation='Exploratory: n50 much smaller than2048 feature dimensions; rank-deficient covariance; not a clinical or fine-vessel accuracy metric. Not directly comparable to paper FID using different sample/preprocessing protocols.')
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
