"""Two FlowOCT controls: pixel or 2D Haar velocity matching, no auxiliary losses."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from flowoct.model import FlowOCT
from flowoct.wavelets import DWT_2D, IDWT_2D


ARMS = {'07_flowoct_fm_only': 'pixel', '08_flowoct_wavelet_only': 'haar2d'}


class FlowOCTVelocityOnly(FlowOCT):
    """Reuse the author's ROI velocity loss and solvers; remove all three auxiliaries.

    There is no contrastive head and no independent OCT input concatenation.
    The Haar arm initializes its generator identically to the full FlowOCT arm
    under the same seed. Pixel mode changes only the generator's input/output
    channels and bypasses DWT/IDWT; its network processes full-size B-scans.
    """
    def __init__(self, opt, representation):
        nn.Module.__init__(self)
        if representation not in ('pixel', 'haar2d'):
            raise ValueError(representation)
        self.opt, self.representation = opt, representation
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')
        self.isTrain = opt.isTrain
        channels = 4 if representation == 'haar2d' else 1
        self.netG = DiffusionModelUNet(spatial_dims=2, in_channels=channels,
                                      out_channels=channels).to(self.device)
        if representation == 'haar2d':
            self.dwt_2d, self.idwt_2d = DWT_2D('haar'), IDWT_2D('haar')
        self.criterionL2 = nn.MSELoss()
        self.vweight = 10.0
        self.generation_steps = 3
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr,
                                                betas=(opt.beta, 0.999))

    def _velocity(self, state, times, recompute=False):
        encoded = torch.cat(self.dwt_2d(state), dim=1) if self.representation == 'haar2d' else state
        chunk = getattr(self.opt, 'network_chunk', 0) if recompute else 0
        if chunk:
            velocity = torch.cat([
                checkpoint(self.netG, encoded[start:start + chunk], times[start:start + chunk],
                           use_reentrant=False)
                for start in range(0, len(encoded), chunk)], dim=0)
        else:
            velocity = self.netG(encoded, times)
        if self.representation == 'haar2d':
            velocity = self.idwt_2d(*(velocity[:, i:i + 1] for i in range(4)))
        return velocity

    def forward(self):
        b, c, d, h, w = self.real_A.shape
        count = getattr(self.opt, 'train_slices', 0) or max(1, d // 4)
        if not 1 <= count <= d:
            raise ValueError('train_slices must lie in [1, volume slice count]')
        indices = torch.randperm(d, device=self.device)[:count]
        source = self.real_A[:, :, indices].permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        target = self.real_B[:, :, indices].permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        times = torch.rand(len(source), device=self.device)
        t = times[:, None, None, None]
        state, truth = (1 - t) * source + t * target, target - source
        self.optimizer_G.zero_grad(set_to_none=True)
        self.vloss = self._cropped_l2_loss(self._velocity(state, times, True), truth)
        self.loss = self.vweight * self.vloss
        if not torch.isfinite(self.loss):
            raise FloatingPointError('Non-finite velocity-only loss; update not applied')
        self.loss.backward()
        self.optimizer_G.step()
        # These are the actual sampler values, rather than a separate reference t.
        self.last_sampling = {'slice_indices': indices.detach().cpu().tolist(),
                              'slice_times': times.detach().cpu().tolist()}

    def euler_step(self, x_t, t_start, t_end):
        return x_t + (t_end - t_start)[:, None, None, None] * self._velocity(x_t, t_start)

    def get_current_loss(self):
        return {'vloss': float(self.vloss.detach())}
