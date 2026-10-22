"""Stitch velocities at each Euler step, then update one global volume state."""
from itertools import product

import torch
from torch.nn import functional as F


def window_starts(length, patch, overlap):
    stride = max(1, round(patch * (1 - overlap)))
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.inference_mode()
def generate_volume(model, source, patch, steps=8, overlap=0.5, amp=False):
    """source [1,1,Z,Y,X]; keep full states on CPU, transfer one window to model device."""
    if source.ndim != 5 or source.shape[:2] != (1, 1):
        raise ValueError('Full-volume inference supports one single-channel case at a time')
    if steps < 1 or len(patch) != 3 or min(patch) < 8 or not 0 <= overlap < 1:
        raise ValueError('steps>=1, patch dimensions>=8, overlap in [0,1) required')
    device = next(model.parameters()).device
    original = source.shape[2:]
    pad = []
    for n, p in reversed(list(zip(original, patch))):
        pad.extend([0, max(0, p - n)])
    source = F.pad(source.detach().to('cpu', dtype=torch.float32), pad, mode='replicate')
    starts = [window_starts(n, p, overlap) for n, p in zip(source.shape[2:], patch)]
    regions = [(slice(None), slice(None)) + tuple(slice(s, s + p) for s, p in zip(pos, patch))
               for pos in product(*starts)]
    # Positive separable Hann weights reduce boundary effects without leaving uncovered edges.
    axes = [torch.hann_window(p, periodic=False).clamp_min(0.05) for p in patch]
    weight = (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])[None, None]
    counts = torch.zeros_like(source)
    for region in regions:
        counts[region] += weight
    state = source.clone()
    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            velocity = torch.zeros_like(state)
            for region in regions:
                current, condition = state[region].to(device), source[region].to(device)
                time = torch.full((1,), step / steps, device=device)
                with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                    v = model(current, time, condition)
                if not torch.isfinite(v).all():
                    raise FloatingPointError('Nonfinite inference velocity')
                velocity[region] += v.float().cpu() * weight
            state.add_(velocity / counts, alpha=1 / steps)
        # No stepwise clamping: the learned ODE may pass outside the image range.
        state = state[:, :, :original[0], :original[1], :original[2]]
        return state.clamp(0, 1)
    finally:
        model.train(was_training)
