"""Shared time-conditioned 3D U-Net for voxel and Haar coefficient velocities."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .wavelets import haar_dwt3d, validate_wavelet_patch


class TimeBlock(nn.Module):
    def __init__(self, cin, cout, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.time = nn.Linear(time_channels, cout)
        self.norm2 = nn.GroupNorm(4, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(embedding))[:, :, None, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class VelocityUNet3D(nn.Module):
    """Input/output: [B,C,Z,Y,X]. Fixed OCT conditioning is an explicit design choice."""
    def __init__(self, base_channels=16, condition=True, channels=1):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError('base_channels must be a positive multiple of 4')
        if channels < 1:
            raise ValueError('channels must be positive')
        self.channels = channels
        self.condition = condition
        b, tc = base_channels, base_channels * 4
        self.register_buffer('frequencies', torch.exp(-math.log(10000) * torch.arange(32) / 31))
        self.time_mlp = nn.Sequential(nn.Linear(64, tc), nn.SiLU(), nn.Linear(tc, tc))
        self.stem = nn.Conv3d(channels * (2 if condition else 1), b, 3, padding=1)
        self.enc0 = TimeBlock(b, b, tc)
        self.down0 = nn.Conv3d(b, 2 * b, 3, stride=2, padding=1)
        self.enc1 = TimeBlock(2 * b, 2 * b, tc)
        self.down1 = nn.Conv3d(2 * b, 4 * b, 3, stride=2, padding=1)
        self.enc2 = TimeBlock(4 * b, 4 * b, tc)
        self.down2 = nn.Conv3d(4 * b, 8 * b, 3, stride=2, padding=1)
        self.middle = TimeBlock(8 * b, 8 * b, tc)
        self.dec2 = TimeBlock(12 * b, 4 * b, tc)
        self.dec1 = TimeBlock(6 * b, 2 * b, tc)
        self.dec0 = TimeBlock(3 * b, b, tc)
        self.output = nn.Sequential(nn.GroupNorm(4, b), nn.SiLU(), nn.Conv3d(b, channels, 3, padding=1))

    def forward(self, state, time, source):
        if state.ndim != 5 or state.shape[1] != self.channels or state.shape != source.shape:
            raise ValueError(f'state/source must have equal [B,{self.channels},Z,Y,X] shapes')
        if min(state.shape[2:]) < 8 or time.shape != (state.shape[0],):
            raise ValueError('spatial dimensions must be >=8; time must have shape [B]')
        angles = time.float()[:, None] * 1000 * self.frequencies[None]
        embedding = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=1))
        x = torch.cat((state, source), dim=1) if self.condition else state
        a = self.enc0(self.stem(x), embedding)
        b = self.enc1(self.down0(a), embedding)
        c = self.enc2(self.down1(b), embedding)
        h = self.middle(self.down2(c), embedding)
        for skip, block in ((c, self.dec2), (b, self.dec1), (a, self.dec0)):
            h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
            h = block(torch.cat((h, skip), dim=1), embedding)
        return self.output(h)  # Signed velocity: no sigmoid/clamp here.


def make_model(base_channels=16, condition=True, representation='voxel'):
    if representation not in ('voxel', 'haar3d'):
        raise ValueError(f'Unknown representation: {representation}')
    return VelocityUNet3D(base_channels, condition, channels=8 if representation == 'haar3d' else 1)


def flow_loss(model, source, target, time=None, representation='voxel'):
    """Straight paired flow; equal-weight MSE over all voxels or all eight bands."""
    if source.shape != target.shape:
        raise ValueError('OCT and OCTA must have equal shapes')
    if representation == 'haar3d':
        validate_wavelet_patch(source.shape[2:])
        source, target = haar_dwt3d(source), haar_dwt3d(target)
    elif representation != 'voxel':
        raise ValueError(f'Unknown representation: {representation}')
    if time is None:
        time = torch.rand(source.shape[0], device=source.device)
    t = time[:, None, None, None, None]
    state = (1 - t) * source + t * target
    velocity = model(state, time, source)
    return F.mse_loss(velocity.float(), (target - source).float())


def z_mean_projection(volume):
    """Full-depth en-face mean: [B,C,Z,Y,X] -> [B,C,Y,X], accumulated in FP32."""
    if volume.ndim != 5:
        raise ValueError('Z projection requires [B,C,Z,Y,X]')
    return volume.mean(dim=2, dtype=torch.float32)


def z_projection_loss(model, source, target):
    """L1 projection supervision on the source-only one-step endpoint at t=0.

    No target is supplied to the network on this branch. This is an auxiliary
    one-step estimate, not the multi-step ODE result used for validation.
    Keep the estimate unclamped so out-of-range predictions retain gradients.
    """
    if source.shape != target.shape or source.ndim != 5 or source.shape[1] != 1:
        raise ValueError('Projection supervision requires paired [B,1,Z,Y,X] voxels')
    time = torch.zeros(source.shape[0], device=source.device)
    endpoint = source.float() + model(source, time, source).float()
    return F.l1_loss(z_mean_projection(endpoint), z_mean_projection(target))
