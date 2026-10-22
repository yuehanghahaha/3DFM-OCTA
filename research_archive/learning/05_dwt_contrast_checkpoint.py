"""Lesson 5: actual FlowOCT shapes, contrastive masks, and Adam resumption.

This uses synthetic data and an untrained FlowOCT network for shape inspection.
The checkpoint experiment trains a separate scalar linear model on CPU.
No medical experiment or existing checkpoint is modified.
"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "upstream"))
from models.FlowOCT_model import FlowOCT


def inspect_wavelet_and_head():
    # The author's DWT creates CUDA matrices whenever CUDA is available.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
    opt = SimpleNamespace(
        gpu_ids=[0] if use_cuda else [],
        isTrain=True,
        checkpoints_dir=str(PROJECT_ROOT / "artifacts"),
        name="learning_only",
        lr=1e-4,
        beta=0.5,
        continue_train=False,
    )
    model = FlowOCT(opt)
    model.eval()

    print("1. Actual DWT -> channel concatenation -> U-Net -> IDWT")
    print("Untrained network; this section only inspects shapes.")
    with torch.no_grad():
        image = torch.rand(2, 1, 32, 32, device=model.device)
        times = torch.tensor([0.2, 0.7], device=model.device)
        bands = model.dwt_2d(image)
        combined = torch.cat(bands, dim=1)
        predicted_bands = model.netG(combined, times)
        split_bands = [predicted_bands[:, i:i + 1] for i in range(4)]
        predicted_velocity = model.idwt_2d(*split_bands)
        reconstruction = model.idwt_2d(*bands)
        for i in range(4):
            assert torch.equal(combined[:, i:i + 1], bands[i])
        assert predicted_velocity.shape == image.shape
        reconstruction_error = (reconstruction - image).abs().max().item()
        assert reconstruction_error < 1e-5
        print("Input:", tuple(image.shape))
        print("Each band:", [tuple(b.shape) for b in bands])
        print("Concatenated bands:", tuple(combined.shape))
        print("Network output:", tuple(predicted_bands.shape))
        print("Velocity:", tuple(predicted_velocity.shape))
        print("IDWT of original bands max reconstruction error:", reconstruction_error)
        print("Integer index:", tuple(predicted_bands[:, 0].shape))
        print("Slice index:", tuple(predicted_bands[:, 0:1].shape))

        print("\n2. Projection head, one layer at a time")
        features = image
        for layer in model.contrast_projection_head:
            features = layer(features)
            print(type(layer).__name__, "->", tuple(features.shape))
        print("Head parameters:", sum(p.numel() for p in model.contrast_projection_head.parameters()))

    print("\n3. Contrastive masks for two pairs; tiny 2D features for clarity")
    fake = torch.eye(2)
    real = torch.eye(2)
    features = F.normalize(torch.cat([fake, real], dim=0), dim=1)
    similarity = features @ features.T
    pair_count = fake.shape[0]
    ids = torch.arange(pair_count)
    positive = torch.zeros(2 * pair_count, 2 * pair_count, dtype=torch.bool)
    positive[ids, ids + pair_count] = True
    positive[ids + pair_count, ids] = True
    negative = ~(torch.eye(2 * pair_count, dtype=torch.bool) | positive)
    pos_scores = similarity[positive].view(2 * pair_count, 1)
    neg_scores = similarity[negative].view(2 * pair_count, -1)
    logits = torch.cat([pos_scores, neg_scores], dim=1) / 0.1
    labels = torch.zeros(2 * pair_count, dtype=torch.long)
    manual_loss = F.cross_entropy(logits, labels)
    repository_loss = model._contrastive_loss(fake, real)
    swapped_loss = model._contrastive_loss(fake, real.flip(0))
    assert torch.allclose(manual_loss, repository_loss)
    assert swapped_loss > repository_loss
    print("Feature order: fake_0, fake_1, real_0, real_1")
    print("Similarity matrix:\n", similarity)
    print("Positive mask (1=True):\n", positive.int())
    print("Negative mask (1=True):\n", negative.int())
    print("Logits:\n", logits)
    print("Labels:", labels.tolist())
    print("Correct pairing loss:", repository_loss.item())
    print("Swapped pairing loss:", swapped_loss.item())
    print("normalize([3,4]):", F.normalize(torch.tensor([[3.0, 4.0]]), dim=1).tolist())


def one_update(net, optimizer, target):
    optimizer.zero_grad()
    prediction = net(torch.ones(1, 1))
    loss = F.mse_loss(prediction, torch.tensor([[target]]))
    loss.backward()
    optimizer.step()
    return net.weight.item()


def inspect_checkpoint_resumption():
    print("\n4. Same saved weights, different Adam history")
    original = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        original.weight.fill_(0.3)
    original_optimizer = torch.optim.Adam(original.parameters(), lr=0.05)
    for target in (2.0, -1.0, 2.0, -1.0):
        one_update(original, original_optimizer, target)

    checkpoint = {
        "model": deepcopy(original.state_dict()),
        "optimizer": deepcopy(original_optimizer.state_dict()),
        "global_step": 4,
    }
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)

    weight_only = nn.Linear(1, 1, bias=False)
    weight_only.load_state_dict(restored["model"])
    fresh_optimizer = torch.optim.Adam(weight_only.parameters(), lr=0.05)

    full = nn.Linear(1, 1, bias=False)
    full.load_state_dict(restored["model"])
    restored_optimizer = torch.optim.Adam(full.parameters(), lr=0.05)
    restored_optimizer.load_state_dict(restored["optimizer"])

    print("Initial weights:", original.weight.item(), weight_only.weight.item(), full.weight.item())
    print("Fresh Adam parameter states:", len(fresh_optimizer.state))
    print("Restored Adam step:", next(iter(restored_optimizer.state.values()))["step"].item())
    uninterrupted_weight = one_update(original, original_optimizer, 1.5)
    weight_only_next = one_update(weight_only, fresh_optimizer, 1.5)
    full_next = one_update(full, restored_optimizer, 1.5)
    print("Next weight, uninterrupted:", uninterrupted_weight)
    print("Next weight, weights only:", weight_only_next)
    print("Next weight, weights + Adam:", full_next)
    assert uninterrupted_weight == full_next
    assert abs(weight_only_next - full_next) > 1e-6
    print("Both restored models can continue training; only full Adam restoration matches this controlled run.")


if __name__ == "__main__":
    torch.manual_seed(2026)
    inspect_wavelet_and_head()
    inspect_checkpoint_resumption()
    print("\nAll teaching checks passed.")
