"""Lesson 2: verify the author's Haar transform and gradient on CUDA."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'upstream'))
from util.DWT_IDWT_layer import DWT_2D, IDWT_2D

# The author's matrices choose CUDA whenever it is available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(2026)
x = torch.randn(2, 1, 32, 32, device=device, requires_grad=True)
dwt, idwt = DWT_2D('haar'), IDWT_2D('haar')
bands = dwt(x)
reconstruction = idwt(*bands)
error = (x - reconstruction).abs().max().item()
assert error < 1e-5
reconstruction.square().mean().backward()
expected_grad = 2 * x.detach() / x.numel()
grad_error = (x.grad - expected_grad).abs().max().item()
assert grad_error < 1e-5
print('Input:', tuple(x.shape), 'bands:', [tuple(b.shape) for b in bands])
print('Haar reconstruction max error:', error, 'gradient max error:', grad_error)
print('No learned parameters in DWT:', sum(p.numel() for p in dwt.parameters()))
