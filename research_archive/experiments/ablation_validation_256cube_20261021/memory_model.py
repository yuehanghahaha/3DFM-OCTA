"""The same U-Net with activation recomputation; no parameter/loss changes."""
from functools import partial
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from fm3d.model import VelocityUNet3D


def up(block, state, skip, embedding):
    state = F.interpolate(state, size=skip.shape[2:], mode='trilinear', align_corners=False)
    return block(torch.cat((state, skip), dim=1), embedding)


class MemoryUNet(VelocityUNet3D):
    def forward(self, state, time, source):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(state, time, source)
        angles = time.float()[:, None] * 1000 * self.frequencies[None]
        embedding = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=1))
        x = torch.cat((state, source), dim=1) if self.condition else state
        a = checkpoint(lambda x, e: self.enc0(self.stem(x), e), x, embedding, use_reentrant=False)
        b = checkpoint(lambda x, e: self.enc1(self.down0(x), e), a, embedding, use_reentrant=False)
        c = checkpoint(lambda x, e: self.enc2(self.down1(x), e), b, embedding, use_reentrant=False)
        h = checkpoint(lambda x, e: self.middle(self.down2(x), e), c, embedding, use_reentrant=False)
        for skip, block in ((c, self.dec2), (b, self.dec1), (a, self.dec0)):
            h = checkpoint(partial(up, block), h, skip, embedding, use_reentrant=False)
        return checkpoint(self.output, h, use_reentrant=False)


def verify_equivalence():
    torch.manual_seed(20261021)
    original = VelocityUNet3D(4, True)
    memory = MemoryUNet(4, True)
    memory.load_state_dict(original.state_dict())
    source, state = torch.rand(1,1,16,16,16), torch.rand(1,1,16,16,16)
    t = torch.tensor([.37])
    a, b = original(state,t,source), memory(state,t,source)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), (_, q) in zip(original.named_parameters(), memory.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4,msg=name)
    return dict(status='passed', forward_and_all_parameter_gradients_match=True)
