"""Tiny time-conditioned 3D U-Net FM lab on synthetic data; NOT a research baseline."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)

    def forward(self, x, embedding):
        h = self.conv1(x) + self.time_proj(embedding)[:, :, None, None, None]
        return F.silu(self.norm2(self.conv2(F.silu(self.norm1(h)))))


class Tiny3DUNet(nn.Module):
    def __init__(self, explicit_condition=False):
        super().__init__()
        self.explicit_condition = explicit_condition
        self.time_mlp = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32))
        self.encoder = TimeBlock(2 if explicit_condition else 1, 8)
        self.middle = TimeBlock(8, 16)
        self.decoder = TimeBlock(24, 8)
        self.output = nn.Conv3d(8, 1, 1)

    def forward(self, x_t, t, source):
        # This lab uses simple Fourier features; MONAI uses its own timestep embedding.
        time_features = torch.stack([t, torch.sin(2 * torch.pi * t), torch.cos(2 * torch.pi * t)], dim=1)
        embedding = self.time_mlp(time_features)
        x = torch.cat([x_t, source], dim=1) if self.explicit_condition else x_t
        skip = self.encoder(x, embedding)
        h = self.middle(F.avg_pool3d(skip, 2), embedding)
        h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.output(self.decoder(torch.cat([h, skip], dim=1), embedding))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--explicit-condition', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/minimal_3d'))
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    d, h, w = torch.meshgrid(*[torch.linspace(-1, 1, 16, device=device)] * 3, indexing='ij')
    target = torch.exp(-(h.square() + (w - 0.3 * torch.sin(3 * d)).square()) / 0.08)[None, None]
    source = (0.25 + 0.15 * torch.cos(5 * w) + 0.15 * target[0, 0])[None, None]
    model = Tiny3DUNet(args.explicit_condition).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(args.steps):
        model.train()
        t = torch.rand(source.shape[0], device=device)
        t_img = t[:, None, None, None, None]
        x_t = (1 - t_img) * source + t_img * target
        target_velocity = target - source
        pred_velocity = model(x_t, t, source)
        loss = F.mse_loss(pred_velocity, target_velocity)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(step, losses[-1], flush=True)
    model.eval()
    with torch.no_grad():
        x = source.clone()
        for k in range(3):
            t = torch.full((x.shape[0],), k / 3, device=device)
            x = torch.clamp(x + model(x, t, source) / 3, 0, 1)
        generated_mae = F.l1_loss(x, target).item()
        identity_mae = F.l1_loss(source, target).item()
    import numpy as np
    np.savez_compressed(args.output / 'synthetic_prediction.npz', oct=source.cpu().numpy()[0, 0],
                        octa=x.cpu().numpy()[0, 0], octa_gt=target.cpu().numpy()[0, 0])
    report = {'scope': 'One synthetic 16^3 volume overfit demonstration only', 'shape': list(source.shape),
              'explicit_condition': args.explicit_condition, 'steps': args.steps, 'losses': losses,
              'generation_mae': generated_mae, 'identity_mae': identity_mae,
              'params': sum(p.numel() for p in model.parameters()),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'elapsed_seconds': time.perf_counter() - start}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'losses'}, indent=2))


if __name__ == '__main__':
    main()
