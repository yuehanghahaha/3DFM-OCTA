"""Frozen three-level evaluation of saved predictions; never changes training data."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize


SSIM = dict(data_range=1.0, win_size=7, gaussian_weights=False,
            use_sample_covariance=True, K1=0.01, K2=0.03, channel_axis=None)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired, finite arrays required')
    if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
        raise ValueError('Inputs must already be in [0,1]; no implicit clipping')


def quality(a, b):
    validate(a, b)
    e = a.astype(np.float64) - b
    mse = float(np.mean(e * e))
    return dict(mae=float(np.abs(e).mean()), mse=mse, rmse=float(np.sqrt(mse)),
                psnr=float(-10 * np.log10(max(mse, 1e-12))),
                ssim=float(structural_similarity(a, b, **SSIM)))


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def binary_scores(pred, gt):
    """2D overlap and hard clDice; two empty masks score 1, one empty scores 0."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    n, m = int(pred.sum()), int(gt.sum())
    inter = int((pred & gt).sum())
    if not n or not m:
        cd = float(n == m)
    else:
        sp, sg = skeletonize(pred), skeletonize(gt)
        tp = float((sp & gt).sum() / sp.sum())
        ts = float((sg & pred).sum() / sg.sum())
        cd = 2 * tp * ts / (tp + ts) if tp + ts else 0.0
    return dict(dice=2 * inter / (n + m) if n + m else 1.0,
                iou=inter / (n + m - inter) if n + m else 1.0,
                precision=inter / n if n else float(m == 0),
                recall=inter / m if m else float(n == 0), cldice=cd)


def vessel_scores(a, b, annotation):
    validate(a, b)
    if annotation.shape != a.shape:
        raise ValueError('Mask must match official projection orientation')
    weight = np.where(annotation, 1.0, 0.1).astype(np.float32)
    out = {'weighted_' + k: v for k, v in quality(a * weight, b * weight).items()}
    ma, mb = a > a.mean(dtype=np.float64), b > b.mean(dtype=np.float64)
    if any(n % 16 for n in a.shape):
        raise ValueError('VDC requires dimensions divisible by 16; no hidden cropping')
    def density(m):
        h, w = m.shape
        return m.reshape(h // 16, 16, w // 16, 16).mean((1, 3))
    out.update(vde=float(abs(ma.mean() - mb.mean())), vdc=corr(density(ma), density(mb)),
               predicted_density=float(ma.mean()), reference_threshold_density=float(mb.mean()),
               annotation_density=float(annotation.mean()))
    out.update({'pred_vs_annotation_' + k: v for k, v in binary_scores(ma, annotation).items()})
    out.update({'reference_vs_annotation_' + k: v for k, v in binary_scores(mb, annotation).items()})
    out.update({'pred_vs_reference_threshold_' + k: v for k, v in binary_scores(ma, mb).items()})
    return out, ma, mb


def aggregate(rows):
    out = {}
    for key in rows[0]:
        if key in ('case_id', 'patient_id', 'disease'):
            continue
        vals = [r[key] for r in rows if r[key] is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            out[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                            n=len(vals), min=float(min(vals)), max=float(max(vals)))
    return out


def save(output, rows, protocol):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (output / 'metrics.json').write_text(json.dumps(dict(protocol=protocol, cases=rows,
        patient_macro=aggregate(rows)), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def lpips_model(device):
    import lpips
    return lpips.LPIPS(net='alex', version='0.1').eval().to(device)


def perceptual(model, a, b, device):
    import torch
    def tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)[:, None].repeat(1, 3, 1, 1) * 2 - 1
    with torch.inference_mode():
        return model(tensor(a), tensor(b)).flatten().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['projection', 'volume'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--lpips', action='store_true')
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    metadata = [r for r in csv.DictReader(args.manifest.open(encoding='utf-8-sig')) if r['split'] == 'test']
    if len(metadata) != 50 or len({r['patient_id'] for r in metadata}) != 50:
        raise ValueError('This protocol is frozen to the 50 independent 3M test patients')
    source = json.loads((args.input / 'metrics.json').read_text(encoding='utf-8'))
    if {str(r['case_id']) for r in source['cases']} != {r['case_id'] for r in metadata}:
        raise ValueError('Prediction report and fixed manifest IDs differ')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='fm3d_three_levels_v1', mode=args.mode, n=50, range=[0, 1],
        manifest_sha256=sha(args.manifest), ssim=SSIM, aggregation='Within patient first; equal patient macro mean and sample SD',
        reference='Paired Z256 OCTA; raw normalized intensities; no contrast stretch',
        psnr='-10log10(max(MSE,1e-12)); capped at120dB for exact equality',
        bscan='volume[:, y, :] => 304 ZX slices per patient; no resize',
        projection='FULL Z-mean; ILM_OPL and OPL_BM annotated layer maxima; inherited frozen projection protocol',
        vessels='ILM_OPL only; GT_LargeVessel OR GT_Capillary >0; BMP orientation unchanged [X,Y]',
        weighted='Multiply BOTH images by weight1 on annotation,0.1 elsewhere; data_range stays1',
        density='Each image independently thresholded strictly above its global mean; VDE absolute occupancy difference; VDC Pearson over nonoverlapping16x16 occupancy patches',
        topology='Hard2D skeletonize Zhang clDice + Dice/IoU; threshold-derived prediction vs official annotation; NOT a learned vessel segmenter or3D topology result',
        empty_masks='Both empty Dice/IoU/clDice=1; only one empty=0; undefined Pearson=null',
        lpips='AlexNet pretrained ImageNet + LPIPS v0.1 learned weights; grayscale replicatedRGB;[0,1] to[-1,1]; no resize; eval FP32' if args.lpips else None,
        packages={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'scikit-image', 'torch', 'Pillow']})
    model = lpips_model(args.device) if args.lpips else None
    if model is not None:
        import lpips
        protocol['packages'].update({k: importlib.metadata.version(k) for k in ['lpips', 'torchvision']})
        protocol['lpips_weights'] = {
            'alexnet_sha256': sha(Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'),
            'linear_sha256': sha(Path(lpips.__file__).parent / 'weights/v0.1/alex.pth')}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows, hashes = [], {}
    for i, meta in enumerate(metadata):
        cid = meta['case_id']
        row = {k: meta[k] for k in ('case_id', 'patient_id', 'disease')}
        if args.mode == 'projection':
            path = args.input / f'{cid}_projections.npz'
            with np.load(path, allow_pickle=False) as f:
                for slab in ['FULL', 'ILM_OPL', 'OPL_BM']:
                    a, b = f['prediction_' + slab], f['reference_z256_' + slab]
                    row.update({slab + '_' + k: v for k, v in quality(a, b).items()})
                    row[slab + '_pcc'] = corr(a, b)
                    if model is not None:
                        row[slab + '_lpips'] = float(perceptual(model, a[None], b[None], args.device)[0])
                    if slab == 'ILM_OPL':
                        label_paths = [args.labels / name / f'{cid}.bmp' for name in ['GT_LargeVessel', 'GT_Capillary']]
                        masks = [np.array(Image.open(p)) > 0 for p in label_paths]
                        mask = masks[0] | masks[1]
                        vs, mp, mg = vessel_scores(a, b, mask)
                        row.update({'vessel_' + k: v for k, v in vs.items()})
                        for name, img in [('prediction', mp), ('reference', mg), ('annotation', mask)]:
                            d = args.output / 'masks' / name
                            d.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(img.astype(np.uint8) * 255).save(d / f'{cid}.png')
                        hashes[cid + '_labels'] = {str(p): sha(p) for p in label_paths}
        else:
            path = args.input / f'{cid}.npz'
            with np.load(path, allow_pickle=False) as f:
                if f['axis_order'].item() != 'ZYX':
                    raise ValueError('Expected ZYX')
                a, b = f['octa'], f['octa_gt']
            if a.shape != (256, 304, 304):
                raise ValueError('Unexpected volume dimensions')
            row.update({'volume_' + k: v for k, v in quality(a, b).items()})
            scans = [quality(a[:, y, :], b[:, y, :]) for y in range(a.shape[1])]
            row.update({'bscan_mean_' + k: float(np.mean([s[k] for s in scans])) for k in scans[0]})
            if model is not None:
                values = []
                for y in range(0, a.shape[1], 8):
                    values.extend(perceptual(model, a[:, y:y+8].transpose(1, 0, 2),
                                             b[:, y:y+8].transpose(1, 0, 2), args.device).tolist())
                row['bscan_mean_lpips'] = float(np.mean(values))
        hashes[cid] = sha(path)
        rows.append(row)
        save(args.output, rows, protocol)
        print(f'{args.mode} {i+1}/50 {cid}', flush=True)
    (args.output / 'input_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
