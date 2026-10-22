"""Fixed [0,1] image metrics; patients, not patches, are the aggregation unit."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
import torch

from .losses import layer_masks, masked_max


def image_metrics(a, b, ssim=True):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    error = a-b
    x, y = a-a.mean(), b-b.mean()
    denominator = np.sqrt(np.sum(x*x)*np.sum(y*y))
    result = dict(mae=float(np.abs(error).mean()),
                  psnr=float(-10*np.log10(max(float(np.mean(error**2)), 1e-12))),
                  pcc=float(np.sum(x*y)/denominator) if denominator > 1e-15 else None)
    if ssim:
        result['ssim'] = float(structural_similarity(a, b, data_range=1., win_size=7))
    return result


def sample_metrics(prediction, sample, native_depth=640, full_ssim=False):
    target = sample['octa']
    result = {'volume_'+k: v for k, v in image_metrics(prediction, target, full_ssim).items()}
    for name, reducer in (('zmean', np.mean), ('zmax', np.max)):
        result.update({name+'_'+k: v for k, v in image_metrics(reducer(prediction, axis=0),
                                                             reducer(target, axis=0)).items()})
    projections = {}
    if 'layers' in sample:
        masks = layer_masks(torch.from_numpy(sample['layers'])[None], len(target), native_depth)
        for name, mask in zip(('ilm_opl', 'opl_bm'), masks):
            a, valid = masked_max(torch.from_numpy(prediction)[None, None], mask)
            b, _ = masked_max(torch.from_numpy(target)[None, None], mask)
            a, b = a[0, 0].numpy(), b[0, 0].numpy()
            result.update({name+'_'+k: v for k, v in image_metrics(a, b).items()})
            result[name+'_empty_rays'] = int((~valid).sum())
            projections[name] = (a, b)
        if 'vessel_mask' in sample:
            region = sample['vessel_mask'].astype(bool)
            err = np.abs(projections['ilm_opl'][0]-projections['ilm_opl'][1])
            result['ilm_opl_vessel_mae'] = float(err[region].mean()) if region.any() else None
    indices = [target.shape[1]//4, target.shape[1]//2, 3*target.shape[1]//4]
    result['bscan_ssim'] = float(np.mean([image_metrics(prediction[:, y], target[:, y])['ssim'] for y in indices]))
    result['prediction_std'], result['target_std'] = float(prediction.std()), float(target.std())
    return result, projections


def aggregate(records):
    result = {}
    for split in sorted({r['split'] for r in records}):
        selected = [r for r in records if r['split'] == split]
        patients = defaultdict(list)
        for r in selected:
            patients[r['patient_id']].append(r['metrics'])
        def average(rows):
            return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
                    if any(r[k] is not None for r in rows) else None for k in rows[0]}
        per_patient = {p: average(rows) for p, rows in patients.items()}
        result[split] = dict(patients=len(patients), patches=len(selected),
                             metrics=average(list(per_patient.values())), per_patient=per_patient)
    return result


def panel(sample, prediction, projections, path):
    source, target = sample['oct'], sample['octa']
    maps = [(source.mean(0), 'OCT mean', .32), (prediction.mean(0), 'Prediction mean', .32),
            (target.mean(0), 'Target mean', .32),
            (np.abs(prediction.mean(0)-target.mean(0)), 'Mean error', .2)]
    a, b = projections.get('ilm_opl', (prediction.max(0), target.max(0)))
    maps += [(a, 'Pred ILM-OPL max', 1.), (b, 'True ILM-OPL max', 1.),
             (np.abs(a-b), 'Projection error', .5), (sample.get('vessel_mask', b), 'Vessel annotation', 1.)]
    y = target.shape[1]//2
    maps += [(source[:, y], 'OCT B-scan', 1.), (prediction[:, y], 'Prediction B-scan', 1.),
             (target[:, y], 'Target B-scan', 1.), (np.abs(prediction[:, y]-target[:, y]), 'B-scan error', .3)]
    canvas = Image.new('RGB', (1040, 940), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 8), f'{sample["case_id"]} {sample["split"]} {sample.get("patch_id", "full")} | fixed display windows', fill='black', font=font)
    for i, (values, title, scale) in enumerate(maps):
        x, y = 12+(i%4)*258, 40+(i//4)*295
        draw.text((x, y), title, fill='black', font=font)
        tile = Image.fromarray(np.rint(np.clip(values/scale, 0, 1)*255).astype(np.uint8))
        if i < 8:
            tile = tile.resize((242, 242), Image.Resampling.NEAREST)
        else:
            tile.thumbnail((242, 260), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y+24))
    canvas.save(path)

