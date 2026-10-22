"""3D velocity field plus an axial-slice encoder used only by the RDCA loss."""
import torch
from torch import nn

from fm3d.model import make_model
from fm3d.wavelets import haar_dwt3d, haar_idwt3d


class FlowOCT3D(nn.Module):
    def __init__(self, base_channels=16, condition=True, representation='haar3d'):
        super().__init__()
        self.representation = representation
        self.velocity = make_model(base_channels, condition, representation)
        # Instantiate after the velocity model: enabling RDCA never changes its seed.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 128))

    def encode(self, x):
        return haar_dwt3d(x) if self.representation == 'haar3d' else x

    def decode(self, x):
        return haar_idwt3d(x) if self.representation == 'haar3d' else x

    def forward(self, state, time, source):
        return self.velocity(state, time, source)

    def rollout(self, source, steps=8):
        """Differentiable OCT-only Euler integration; no intermediate clipping."""
        if steps < 1:
            raise ValueError('steps must be positive')
        condition = self.encode(source)
        state = condition
        for i in range(steps):
            t = torch.full((len(source),), i / steps, device=source.device)
            state = state.float() + self(state, t, condition).float() / steps
        return self.decode(state)

