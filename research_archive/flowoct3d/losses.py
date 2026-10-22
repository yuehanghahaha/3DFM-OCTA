"""Explicit 3D adaptations of FM, reconstruction, RDCA and structure objectives.

Axes are always B,C,Z,Y,X. Layer annotations are loss targets, never model inputs.
The path endpoint follows the released FlowOCT training implementation; it is NOT
the OCT-only free-running prediction used by validation and deployment.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    fm_weight: float = 1.0
    reconstruction_weight: float = 0.5
    projection_weight: float = 0.1
    rdca_weight: float = 0.005
    structure: bool = True
    mask_ratio: float = 0.05
    background_weight: float = 0.1
    temperature: float = 0.1
    rdca_slices: int = 32
    projection: str = 'retinal'
    auxiliary: str = 'path'
    native_depth: int = 640

    def __post_init__(self):
        weights = (self.fm_weight, self.reconstruction_weight,
                   self.projection_weight, self.rdca_weight)
        if any(not math.isfinite(v) or v < 0 for v in weights) or not any(weights):
            raise ValueError('Loss weights must be finite, nonnegative and not all zero')
        if not 0 < self.mask_ratio < 1 or not 0 <= self.background_weight <= 1:
            raise ValueError('Invalid structure mask ratio/background weight')
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.rdca_slices < 2:
            raise ValueError('RDCA requires temperature>0 and at least two slices')
        if self.projection not in ('retinal', 'full_z') or self.auxiliary not in ('path', 'source'):
            raise ValueError('Unknown projection or auxiliary endpoint')
        if self.native_depth < 1:
            raise ValueError('native_depth must be positive')

    def to_dict(self):
        return asdict(self)


def structure_mask(target, ratio):
    maximum = target.detach().flatten(2).amax(-1)[:, :, None, None, None]
    return (target.detach() > ratio * maximum).float()


def weighted_mean(error, support, background_weight):
    """Normalize per patient, then average patients; empty masks use full loss."""
    if support is None:
        return error.flatten(1).mean(1).mean()
    weights = background_weight + (1 - background_weight) * support
    weights = weights.expand_as(error)
    denominator = weights.flatten(1).sum(1)
    numerator = (error * weights).flatten(1).sum(1)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12),
                       error.flatten(1).mean(1)).mean()


def layer_masks(layers, depth, native_depth=640):
    """MATLAB 1-based ILM/OPL/BM boundaries -> masks on the resampled Z grid.

layers: B,6,Y,X; order ILM,IPL,OPL,ISOS,RPE,BM. Inclusive endpoints match
the existing evaluation protocol. Crops must span the full resampled Z depth.
"""
    if layers.ndim != 4 or layers.shape[1] != 6 or not torch.isfinite(layers).all():
        raise ValueError('layers must be finite [B,6,Y,X] in native 1-based coordinates')
    if (layers < 1).any() or (layers > native_depth).any() or (layers[:, 1:] < layers[:, :-1]).any():
        raise ValueError('Invalid or unordered retinal layer coordinates')
    centers = ((torch.arange(depth, device=layers.device, dtype=torch.float32) + .5)
               * native_depth / depth - .5)[None, :, None, None]
    return [((centers >= layers[:, lo:lo+1] - 1) &
             (centers <= layers[:, hi:hi+1] - 1))[:, None]
            for lo, hi in ((0, 2), (2, 5))]


def masked_max(volume, region):
    """Negative raw predictions remain negative; empty rays have value zero."""
    valid = region.any(dim=2)
    projected = volume.float().masked_fill(~region, -torch.inf).amax(dim=2)
    return torch.where(valid, projected, torch.zeros_like(projected)), valid


def projection_loss(prediction, target, regions, support, background_weight):
    losses = []
    for region in regions:
        a, valid = masked_max(prediction, region)
        b, _ = masked_max(target, region)
        weights = valid.float()
        if support is not None:
            signal = (support.bool() & region).any(dim=2).float()
            weights = weights * (background_weight + (1 - background_weight) * signal)
        # Invalid rays must stay excluded even when background_weight > 0.
        numerator = ((a-b).abs() * weights).flatten(1).sum(1)
        denominator = weights.flatten(1).sum(1)
        losses.append((numerator / denominator.clamp_min(1)).mean())
    return torch.stack(losses).mean()


def rdca_loss(encoder, prediction, target, support, config):
    """Eq.15-style predicted->real InfoNCE within each patient's axial planes.

Batch 1 still has multiple depth negatives. Samples/patients are never collapsed
into a projection axis. Both branches share the trainable depth encoder.
"""
    values, used = [], []
    for b in range(len(prediction)):
        occupied = (target[b, 0].detach().abs().flatten(1).sum(1) > 0).nonzero().flatten()
        if len(occupied) < 2:
            continue
        count = min(config.rdca_slices, len(occupied))
        z = occupied[torch.linspace(0, len(occupied)-1, count, device=occupied.device).round().long()]
        a, y = prediction[b, :, z].permute(1, 0, 2, 3), target[b, :, z].permute(1, 0, 2, 3)
        if support is not None:
            m = support[b, :, z].permute(1, 0, 2, 3)
            # Soft context retains some background signal as in the voxel weights.
            m = config.background_weight + (1-config.background_weight) * m
            a, y = a * m, y * m
        # Dot products/logsumexp stay FP32 under AMP.
        fa = F.normalize(encoder(a).float(), dim=1)
        fy = F.normalize(encoder(y).float(), dim=1)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            logits = fa @ fy.T / config.temperature
            values.append(F.cross_entropy(logits, torch.arange(count, device=logits.device)))
        used.append(count)
    zero = prediction.sum() * 0
    return (torch.stack(values).mean() if values else zero), sum(used)


def objectives(model, source, target, time, config, layers=None):
    if source.shape != target.shape or source.ndim != 5 or source.shape[1] != 1:
        raise ValueError('Paired [B,1,Z,Y,X] source/target required')
    if time.shape != (len(source),) or not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
        raise ValueError('time must be a finite [B] tensor in [0,1]')
    condition, truth = model.encode(source), model.encode(target)
    fraction = time[:, None, None, None, None]
    state = (1-fraction)*condition + fraction*truth
    velocity = model(state, time, condition).float()
    support = structure_mask(target, config.mask_ratio) if config.structure else None
    frequency_support = (F.avg_pool3d(support, 2) if support is not None and
                         model.representation == 'haar3d' else support)
    fm = weighted_mean((velocity-(truth-condition)).square(), frequency_support,
                       config.background_weight)
    zero = fm.new_zeros(())
    reconstruction, projection, rdca, slices = zero, zero, zero, 0
    if config.reconstruction_weight or config.projection_weight or config.rdca_weight:
        if config.auxiliary == 'path':
            prediction = model.decode(state + (1-fraction)*velocity)
        else:
            t0 = torch.zeros_like(time)
            prediction = model.decode(condition + model(condition, t0, condition).float())
        # Do not clamp training endpoints: out-of-range values must keep gradients.
        if config.reconstruction_weight:
            reconstruction = weighted_mean((prediction-target).abs(), support, config.background_weight)
        if config.projection_weight:
            if config.projection == 'retinal':
                if layers is None or layers.shape != (len(target), 6, *target.shape[-2:]):
                    raise ValueError('Retinal projection supervision requires matching layer annotations')
                regions = layer_masks(layers, target.shape[2], config.native_depth)
            else:
                regions = [torch.ones_like(target, dtype=torch.bool)]
            projection = projection_loss(prediction, target, regions, support, config.background_weight)
        if config.rdca_weight:
            rdca, slices = rdca_loss(model.depth_encoder, prediction, target, support, config)
    total = (config.fm_weight*fm + config.reconstruction_weight*reconstruction +
             config.projection_weight*projection + config.rdca_weight*rdca)
    parts = dict(loss=total, fm=fm, reconstruction=reconstruction, projection=projection,
                 rdca=rdca, foreground_fraction=support.mean() if support is not None else fm.new_ones(()),
                 rdca_planes=fm.new_tensor(float(slices)))
    return total, parts

