"""Differentiable, orthonormal, single-level 3D Haar transforms in Z,Y,X order.

No learned compression, band dropping, coefficient clipping or extra dependency.
High pass is (even - odd)/sqrt(2), matching PyWavelets' Haar convention.
"""
import math

import torch


SUBBANDS = ('LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH')
INV_SQRT2 = 1 / math.sqrt(2)


def validate_wavelet_patch(patch):
    """Arguments describe voxel patches; the network sees half of each dimension."""
    if len(patch) != 3 or any(p < 16 or p % 2 for p in patch):
        raise ValueError('haar3d requires even voxel patch dimensions >=16 (coefficient dimensions >=8)')


def haar_dwt3d(volume):
    """[B,C,Z,Y,X] -> [B,8*C,Z/2,Y/2,X/2], band-major channels.

    Spatial sizes must be even. Whole-volume inference pads odd edges once before
    transforming; training uses even patches. Half inputs are promoted for precision.
    """
    if volume.ndim != 5 or min(volume.shape) < 1 or any(n % 2 for n in volume.shape[2:]):
        raise ValueError('Haar DWT expects nonempty [B,C,Z,Y,X] with even spatial dimensions')
    if not volume.is_floating_point():
        raise ValueError('Haar DWT expects floating point intensities')
    if volume.dtype in (torch.float16, torch.bfloat16):
        volume = volume.float()
    bands = [volume]
    for axis in (2, 3, 4):
        transformed = []
        for band in bands:
            even_slice, odd_slice = [slice(None)] * 5, [slice(None)] * 5
            even_slice[axis], odd_slice[axis] = slice(0, None, 2), slice(1, None, 2)
            even, odd = band[tuple(even_slice)], band[tuple(odd_slice)]
            transformed.extend(((even + odd) * INV_SQRT2, (even - odd) * INV_SQRT2))
        bands = transformed
    return torch.cat(bands, dim=1)


def haar_idwt3d(coefficients):
    """Inverse of haar_dwt3d; returns the even, padded voxel grid without clipping."""
    if coefficients.ndim != 5 or min(coefficients.shape) < 1 or coefficients.shape[1] % 8:
        raise ValueError('Haar IDWT expects nonempty [B,8*C,Z,Y,X]')
    if not coefficients.is_floating_point():
        raise ValueError('Haar IDWT expects floating point coefficients')
    if coefficients.dtype in (torch.float16, torch.bfloat16):
        coefficients = coefficients.float()
    bands = list(coefficients.chunk(8, dim=1))
    for axis in (4, 3, 2):
        reconstructed = []
        for index in range(0, len(bands), 2):
            low, high = bands[index:index + 2]
            even, odd = (low + high) * INV_SQRT2, (low - high) * INV_SQRT2
            reconstructed.append(torch.stack((even, odd), dim=axis + 1).flatten(axis, axis + 1))
        bands = reconstructed
    return bands[0]
