"""CPU lesson: self, shapes, flow endpoints, ROI gradients, and InfoNCE.

Run in the existing flowoct environment:
    python learning/04_python_and_flowoct.py

These small synthetic examples explain code behavior; they do not train FlowOCT.
"""

import torch
import torch.nn.functional as F


class Experiment:
    def __init__(self, run_name):
        self.run_name = run_name
        self.steps = 0

    def record_step(self):
        self.steps += 1
        message = f"{self.run_name}: step {self.steps}"
        return message


def slice_contrastive_loss(feat_fake, feat_real, temperature=0.1):
    """Same positive/negative arrangement as the pasted model, on CPU."""
    count = feat_fake.shape[0]
    features = F.normalize(torch.cat([feat_fake, feat_real], dim=0), dim=1)
    similarity = features @ features.T
    indices = torch.arange(count, device=features.device)
    positive_mask = torch.zeros(
        2 * count, 2 * count, dtype=torch.bool, device=features.device
    )
    positive_mask[indices, indices + count] = True
    positive_mask[indices + count, indices] = True
    negative_mask = ~(
        torch.eye(2 * count, dtype=torch.bool, device=features.device)
        | positive_mask
    )
    positive = similarity[positive_mask].view(2 * count, 1)
    negative = similarity[negative_mask].view(2 * count, -1)
    logits = torch.cat([positive, negative], dim=1) / temperature
    labels = torch.zeros(2 * count, dtype=torch.long, device=features.device)
    return F.cross_entropy(logits, labels), tuple(logits.shape)


def main():
    torch.manual_seed(2026)

    print("1. self refers to the object used for this call")
    first = Experiment("baseline")
    second = Experiment("ablation")
    print(first.record_step())
    print(Experiment.record_step(first))  # Explicitly passing the same object.
    print("Independent steps:", first.steps, second.steps)
    print("Local message stored on object:", hasattr(first, "message"))

    print("\n2. Volume -> paired slices -> 2D batch")
    oct_volume = torch.rand(2, 1, 8, 32, 32)
    octa_volume = torch.rand_like(oct_volume)
    batch, channels, depth, height, width = oct_volume.shape
    count = max(1, depth // 4)
    indices = torch.randperm(depth)[:count]
    oct_slices = oct_volume[:, :, indices]
    octa_slices = octa_volume[:, :, indices]
    oct_flat = oct_slices.permute(0, 2, 1, 3, 4).reshape(
        -1, channels, height, width
    )
    octa_flat = octa_slices.permute(0, 2, 1, 3, 4).reshape_as(oct_flat)
    print("Volume:", tuple(oct_volume.shape))
    print("Paired indices:", indices.tolist())
    print("Selected slices:", tuple(oct_slices.shape))
    print("Flattened batch:", tuple(oct_flat.shape))

    print("\n3. Broadcasting and the exact endpoint identity")
    t = torch.rand(oct_flat.shape[0])
    t_img = t.view(-1, 1, 1, 1)
    x_t = (1 - t_img) * oct_flat + t_img * octa_flat
    target_velocity = octa_flat - oct_flat
    perfect_endpoint = x_t + (1 - t_img) * target_velocity
    endpoint_error = (perfect_endpoint - octa_flat).abs().max().item()
    assert endpoint_error < 1e-6
    print("Time shapes:", tuple(t.shape), "->", tuple(t_img.shape))
    print("Exact-velocity endpoint max error:", endpoint_error)

    print("\n4. Euler updates change the image; optimizer updates parameters")
    x = oct_flat.clone()
    for _ in range(3):
        x = x + target_velocity / 3
    euler_error = (x - octa_flat).abs().max().item()
    assert euler_error < 1e-6
    print("Three oracle Euler steps max error:", euler_error)
    print("Oracle velocity uses GT and is only available in this lesson.")

    weight = torch.nn.Parameter(torch.tensor(0.1))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    optimizer.zero_grad()
    loss = F.mse_loss(weight * x_t, target_velocity)
    loss.backward()
    before = weight.item()
    gradient = weight.grad.item()
    optimizer.step()
    print("Parameter gradient:", gradient)
    print("Parameter update:", before, "->", weight.item())

    print("\n5. Projection ROI indexing and gradients through a crop")
    projection = torch.ones(1, height, width)
    xm, ym, xM, yM = 3, 4, 10, 11
    bad_roi = projection[ym:yM + 1, xm:xM + 1]
    good_roi = projection[:, ym:yM + 1, xm:xM + 1]
    print("Wrong indexing shape:", tuple(bad_roi.shape))
    print("Wrong empty-ROI mean:", F.l1_loss(bad_roi, bad_roi).item())
    print("Correct indexing shape:", tuple(good_roi.shape))
    assert bad_roi.numel() == 0 and tuple(good_roi.shape) == (1, 8, 8)

    pred = torch.ones(1, 1, height, width, requires_grad=True)
    detached_array = pred[0].detach().cpu().numpy().squeeze(0)
    assert detached_array.shape == (height, width)
    cropped_loss = pred[:, :, ym:yM + 1, xm:xM + 1].square().mean()
    cropped_loss.backward()
    print("ROI gradient:", pred.grad[0, 0, ym, xm].item())
    print("Outside-ROI gradient:", pred.grad[0, 0, 0, 0].item())
    assert pred.grad[0, 0, ym, xm] > 0 and pred.grad[0, 0, 0, 0] == 0

    print("\n6. Flattening patients before projection can mix their data")
    per_volume = octa_flat.reshape(batch, count, channels, height, width).amax(dim=1)
    mixed = octa_flat.max(dim=0)[0]
    print("Per-volume projection:", tuple(per_volume.shape))
    print("Original mixed projection:", tuple(mixed.shape))

    print("\n7. InfoNCE: number of slice pairs determines the negatives")
    for pair_count in (1, 4):
        fake_features = torch.randn(pair_count, 128, requires_grad=True)
        real_features = torch.randn(pair_count, 128, requires_grad=True)
        contrast_loss, logits_shape = slice_contrastive_loss(fake_features, real_features)
        contrast_loss.backward()
        print(
            "Pairs:", pair_count, "logits:", logits_shape,
            "loss:", contrast_loss.item(),
            "fake feature gradient norm:", fake_features.grad.norm().item(),
        )
        if pair_count == 1:
            assert contrast_loss.item() == 0.0

    print("\nAll numerical teaching checks passed (CPU, synthetic data).")


if __name__ == "__main__":
    main()
