"""Lesson 1: shapes, broadcast, and train/inference distinction."""
import torch

torch.manual_seed(2026)
oct_volume = torch.rand(1, 1, 16, 32, 32)
octa_volume = torch.rand_like(oct_volume)
indices = torch.tensor([1, 4, 9, 12])
oct_slices = oct_volume[:, :, indices]
octa_slices = octa_volume[:, :, indices]
oct_flat = oct_slices.permute(0, 2, 1, 3, 4).reshape(-1, 1, 32, 32)
octa_flat = octa_slices.permute(0, 2, 1, 3, 4).reshape_as(oct_flat)
t = torch.rand(oct_flat.shape[0])
t_img = t[:, None, None, None]
x_t = (1 - t_img) * oct_flat + t_img * octa_flat
target_v = octa_flat - oct_flat
perfect_endpoint = x_t + (1 - t_img) * target_v
assert torch.allclose(perfect_endpoint, octa_flat, atol=1e-6)
for name in ['oct_volume', 'oct_slices', 'oct_flat', 't', 't_img', 'x_t', 'target_v']:
    print(name, tuple(globals()[name].shape))
print('Perfect velocity recovers endpoint; error:', (perfect_endpoint - octa_flat).abs().max().item())

# A learnable scalar stands in for the neural network; the optimizer changes parameters.
scale = torch.nn.Parameter(torch.tensor(0.1))
optimizer = torch.optim.SGD([scale], lr=0.1)
optimizer.zero_grad()
pred_v = scale * x_t
loss = torch.nn.functional.mse_loss(pred_v, target_v)
loss.backward()
print('loss:', loss.item(), 'parameter gradient:', scale.grad.item())
old = scale.item()
optimizer.step()
print('Parameter update:', old, '->', scale.item())

# Oracle velocity is only available in this demonstration, never in deployment.
x = oct_flat.clone()
for step in range(3):
    x = x + target_v / 3
assert torch.allclose(x, octa_flat, atol=1e-6)
print('3 Euler updates using oracle velocity recover the endpoint.')
