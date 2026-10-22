"""Small-patient ablations. All generated-image losses start from OCT alone."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fm3d.model import make_model
from fm3d.wavelets import haar_dwt3d, haar_idwt3d


ARMS = {
    '01_fm3d': ('voxel', 0.0, 0.0),
    '02_fm3d_wavelet': ('haar3d', 0.0, 0.0),
    '03_fm3d_zproj': ('voxel', 0.1, 0.0),
    '04_fm3d_wavelet_zproj': ('haar3d', 0.1, 0.0),
    '05_fm3d_wavelet_zproj_reconstruction': ('haar3d', 0.1, 1.0),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(volume, representation):
    if representation == 'voxel':
        return volume
    if representation == 'haar3d':
        return haar_dwt3d(volume)
    raise ValueError(representation)


def decode(state, representation):
    return haar_idwt3d(state) if representation == 'haar3d' else state


def rollout(model, source, representation, steps=8, checkpoint_steps=False):
    """Differentiable free Euler rollout. NO target or target-derived input."""
    condition = encode(source, representation)
    state = condition
    for index in range(steps):
        time = torch.full((len(source),), index / steps, device=source.device)
        if checkpoint_steps and torch.is_grad_enabled():
            velocity = checkpoint(model, state, time, condition, use_reentrant=False)
        else:
            velocity = model(state, time, condition)
        state = state.float() + velocity.float() / steps
    return decode(state, representation)  # Unclamped during training: preserve gradients.


def objectives(model, source, target, time, arm, steps, checkpoint_steps=False):
    representation = arm['representation']
    condition, truth = encode(source, representation), encode(target, representation)
    fraction = time[:, None, None, None, None]
    state = (1 - fraction) * condition + fraction * truth
    velocity = model(state, time, condition)
    fm = F.mse_loss(velocity.float(), (truth - condition).float())
    projection, reconstruction = fm.new_zeros(()), fm.new_zeros(())
    if arm['lambda_projection'] or arm['lambda_reconstruction']:
        generated = rollout(model, source, representation, steps, checkpoint_steps)
        if arm['lambda_projection']:
            projection = F.l1_loss(generated.mean(2, dtype=torch.float32),
                                   target.mean(2, dtype=torch.float32))
        if arm['lambda_reconstruction']:
            reconstruction = F.l1_loss(generated.float(), target.float())
    total = fm + arm['lambda_projection'] * projection + arm['lambda_reconstruction'] * reconstruction
    return total, {'fm_loss': fm, 'z_projection_loss': projection,
                   'reconstruction_loss': reconstruction, 'loss': total}


def load_samples(root, selection):
    samples = []
    for record in selection['patches']:
        path = root / record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Cached input changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            sample = {key: data[key].copy() for key in ('oct', 'octa', 'layers', 'vessel_mask')}
        sample.update(record)
        samples.append(sample)
    return samples


def layer_projection(volume, layers):
    """ILM..OPL max on full-Z patches; native layer coordinates are MATLAB 1-based."""
    centers = ((np.arange(volume.shape[0]) + .5) * 640 / volume.shape[0] - .5)[:, None, None]
    mask = (centers >= layers[0] - 1) & (centers <= layers[2] - 1)
    return np.where(mask, volume, 0).max(0), int((mask.sum(0) == 0).sum())


def pcc(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return float(np.sum(a*b) / denominator) if denominator > 1e-15 else None


def image_metrics(a, b):
    error = a.astype(np.float64) - b
    return {'mae': float(np.abs(error).mean()),
            'psnr': float(-10 * np.log10(max(float(np.mean(error**2)), 1e-12))),
            'ssim': float(structural_similarity(a, b, data_range=1., win_size=7)),
            'pcc': pcc(a, b)}


def mask_metrics(prediction, reference):
    prediction, reference = prediction.astype(bool), reference.astype(bool)
    a, b = int(prediction.sum()), int(reference.sum())
    intersection = int((prediction & reference).sum())
    dice = 2 * intersection / (a+b) if a+b else 1.
    if a == 0 or b == 0:
        cldice = float(a == b)
    else:
        sp, sg = skeletonize(prediction), skeletonize(reference)
        precision = float((sp & reference).sum() / sp.sum())
        sensitivity = float((sg & prediction).sum() / sg.sum())
        cldice = 2 * precision * sensitivity / (precision+sensitivity) if precision+sensitivity else 0.
    return {'dice': dice, 'cldice': cldice}


def sample_metrics(prediction, sample):
    target = sample['octa']
    err = prediction.astype(np.float64) - target
    result = {'volume_mae': float(np.abs(err).mean()),
              'volume_psnr': float(-10 * np.log10(max(float(np.mean(err**2)), 1e-12)))}
    # Weight patches equally within each patient, then weight patients equally.
    result.update({'zmean_'+k: v for k, v in image_metrics(prediction.mean(0), target.mean(0)).items()})
    a, empty = layer_projection(prediction, sample['layers'])
    b, _ = layer_projection(target, sample['layers'])
    result.update({'ilm_opl_'+k: v for k, v in image_metrics(a, b).items()})
    annotation = sample['vessel_mask'].astype(bool)
    result['ilm_opl_vessel_mae'] = float(np.abs(a-b)[annotation].mean()) if annotation.any() else None
    background = ~annotation
    result['ilm_opl_background_mae'] = float(np.abs(a-b)[background].mean()) if background.any() else None
    result['ilm_opl_empty_rays'] = empty
    # A transparent threshold proxy, not an actual trained segmentation result.
    result.update({'threshold_proxy_'+k: v for k, v in mask_metrics(a > a.mean(dtype=np.float64), annotation).items()})
    result['bscan_ssim'] = float(np.mean([structural_similarity(prediction[:, y, :], target[:, y, :],
                                      data_range=1., win_size=7) for y in (target.shape[1]//4, target.shape[1]//2, 3*target.shape[1]//4)]))
    result['prediction_std'] = float(prediction.std())
    result['target_std'] = float(target.std())
    return result, a, b


def aggregate(records):
    out = {}
    for split in ('train', 'validation'):
        subset = [r for r in records if r['split'] == split]
        patients = defaultdict(list)
        for record in subset:
            patients[record['patient_id']].append(record['metrics'])
        keys = subset[0]['metrics'] if subset else []
        patient_scores = {}
        for patient, patches in patients.items():
            patient_scores[patient] = {key: float(np.mean([p[key] for p in patches if p[key] is not None]))
                                      if any(p[key] is not None for p in patches) else None for key in keys}
        macro = {key: float(np.mean([r[key] for r in patient_scores.values() if r[key] is not None]))
                 if any(r[key] is not None for r in patient_scores.values()) else None for key in keys}
        out[split] = {'patients': len(patients), 'patches': len(subset), 'metrics': macro,
                      'per_patient': patient_scores}
    return out


def font(size=18):
    for name in ('DejaVuSans.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def panel(sample, prediction, projection, truth, destination):
    source = sample['oct']
    reference = sample['octa']
    maps = [(source.mean(0), 80/255), (prediction.mean(0), 80/255), (reference.mean(0), 80/255),
            (np.abs(prediction.mean(0)-reference.mean(0)), .2),
            (projection, 1.), (truth, 1.), (np.abs(projection-truth), .5),
            (sample['vessel_mask'], 1.),
            (source[:, source.shape[1]//2, :], 1.), (prediction[:, source.shape[1]//2, :], 1.), (reference[:, source.shape[1]//2, :], 1.),
            (np.abs(prediction[:, source.shape[1]//2, :]-reference[:, source.shape[1]//2, :]), .3)]
    titles = ['OCT Z mean', 'Generated Z mean', 'Real OCTA Z mean', 'Z mean abs error',
              'Generated ILM-OPL', 'Real ILM-OPL', 'ILM-OPL abs error', 'Official vessel mask',
              'OCT B-scan', 'Generated B-scan', 'Real OCTA B-scan', 'B-scan abs error']
    canvas = Image.new('RGB', (1050, 960), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'{sample["case_id"]} {sample["split"]} {sample["patch_id"]} | fixed windows; no per-image stretch', fill='black', font=font())
    for index, ((values, maximum), title) in enumerate(zip(maps, titles)):
        y, x = 44 + index//4*300, 12 + index%4*260
        draw.text((x, y), title, fill='black', font=font(16))
        tile = Image.fromarray(np.rint(np.clip(values/maximum, 0, 1)*255).astype(np.uint8))
        # B-scans retain their Z:X aspect ratio, although physical voxel spacing differs.
        if index >= 8:
            tile.thumbnail((248, 260), Image.Resampling.NEAREST)
        else:
            tile = tile.resize((248, 248), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y+25))
    canvas.save(destination)


@torch.inference_mode()
def evaluate(model, samples, arm, config, destination=None):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    records = []
    try:
        for sample in samples:
            source = torch.from_numpy(sample['oct'])[None, None].to(device)
            with torch.autocast(device_type=device.type, enabled=config['amp'] and device.type == 'cuda'):
                raw = rollout(model, source, arm['representation'], config['ode_steps'])
            if not torch.isfinite(raw).all():
                raise FloatingPointError('Nonfinite free-rollout prediction')
            prediction = raw.float().clamp(0, 1)[0, 0].cpu().numpy()
            metrics, projection, truth = sample_metrics(prediction, sample)
            metrics['raw_outside_range_fraction'] = float(((raw < 0) | (raw > 1)).float().mean().item())
            records.append({k: sample[k] for k in ('case_id', 'patient_id', 'split', 'disease', 'patch_id')})
            records[-1]['metrics'] = metrics
            if destination is not None:
                destination.mkdir(parents=True, exist_ok=True)
                name = sample['case_id']+'_'+sample['patch_id']
                np.savez_compressed(destination / (name+'.npz'), octa=prediction, octa_gt=sample['octa'],
                                    axis_order=np.array('ZYX'))
                panel(sample, prediction, projection, truth, destination / (name+'.png'))
    finally:
        model.train(was_training)
    return {'cases': records, 'summary': aggregate(records)}


def schedule(seed, updates, count):
    """Balanced independent schedule: model RNG consumption cannot alter crops/t."""
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.permutation(count) for _ in range(math.ceil(updates/count))])[:updates]
    return indices, rng.random(updates).astype(np.float32)


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
