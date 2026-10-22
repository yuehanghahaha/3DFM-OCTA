import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fm3d.inference import generate_volume
from fm3d.wavelets import haar_dwt3d, haar_idwt3d
from .losses import (LossConfig, layer_masks, masked_max, objectives,
                     projection_loss, rdca_loss, structure_mask, weighted_mean)
from .model import FlowOCT3D
from .study import arms, schedule
from .utils import apply_update, save_checkpoint, seed_all


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        seed_all(42)
        self.source, self.target = torch.rand(2, 1, 32, 16, 16), torch.rand(2, 1, 32, 16, 16)
        self.layers = torch.tensor([80, 160, 240, 320, 400, 560.]).view(1, 6, 1, 1).expand(2, 6, 16, 16)

    def test_haar_inverse_energy_gradient(self):
        x = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64, requires_grad=True)
        w = haar_dwt3d(x)
        torch.testing.assert_close(haar_idwt3d(w), x)
        torch.testing.assert_close(w.square().sum(), x.square().sum())
        haar_idwt3d(w).square().sum().backward()
        torch.testing.assert_close(x.grad, 2*x.detach())

    def test_exact_velocity_oracle(self):
        class Oracle(FlowOCT3D):
            def forward(inner, state, time, source):
                return inner.encode(self.target)-source
        model = Oracle(4)
        config = replace(LossConfig(), rdca_weight=0.)
        loss, parts = objectives(model, self.source, self.target, torch.tensor([0., .7]), config, self.layers)
        self.assertLess(float(loss), 1e-6)
        self.assertLess(float(parts['projection']), 1e-6)

    def test_each_module_backpropagates(self):
        model = FlowOCT3D(4)
        for term in ('fm_weight', 'reconstruction_weight', 'projection_weight', 'rdca_weight'):
            weights = dict(fm_weight=0., reconstruction_weight=0., projection_weight=0., rdca_weight=0.)
            weights[term] = 1.
            model.zero_grad(set_to_none=True)
            loss, parts = objectives(model, self.source, self.target, torch.tensor([.2, .6]),
                                     LossConfig(**weights), self.layers)
            loss.backward()
            norm = sum(float(p.grad.abs().sum()) for p in model.velocity.parameters() if p.grad is not None)
            self.assertGreater(norm, 0., term)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if term == 'rdca_weight':
                self.assertEqual(float(parts['rdca_planes']), 64.)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.depth_encoder.parameters()), 0.)

    def test_rdca_pairs_depth_batch_one_and_empty(self):
        config = replace(LossConfig(), rdca_slices=4)
        planes = torch.eye(4).reshape(1, 1, 4, 2, 2)
        encoder = nn.Flatten()
        matched, n = rdca_loss(encoder, planes, planes, None, config)
        permuted, _ = rdca_loss(encoder, planes.flip(2), planes, None, config)
        self.assertEqual(n, 4)
        self.assertLess(float(matched), float(permuted))
        empty, n = rdca_loss(encoder, planes, torch.zeros_like(planes), None, config)
        self.assertEqual(n, 0)
        self.assertEqual(float(empty), 0.)

    def test_mask_and_weighted_reduction(self):
        target = torch.tensor([0., .01, .5, 1.]).view(1, 1, 1, 1, 4)
        mask = structure_mask(target, .05)
        self.assertEqual(mask.flatten().tolist(), [0., 0., 1., 1.])
        error = torch.tensor([10., 10., 2., 4.]).view_as(target)
        self.assertAlmostEqual(float(weighted_mean(error, mask, 0.)), 3.)
        self.assertAlmostEqual(float(weighted_mean(error, mask, 1.)), 6.5)
        self.assertTrue(torch.isfinite(weighted_mean(error, torch.zeros_like(mask), 0.)))

    def test_layer_projection_batch_and_negative_values(self):
        region = layer_masks(self.layers, 32)[0]
        volume = torch.full_like(self.source, -2.)
        projection, valid = masked_max(volume, region)
        self.assertTrue(valid.all())
        torch.testing.assert_close(projection, torch.full_like(projection, -2.))
        a, b = self.target.clone(), self.target.clone()
        a[1] += .25
        loss = projection_loss(a, b, [region], None, .1)
        self.assertAlmostEqual(float(loss), .125, places=6)
        empty, valid = masked_max(volume, torch.zeros_like(region))
        self.assertEqual(float(empty.sum()), 0.)
        self.assertFalse(valid.any())

    def test_shape_and_missing_labels_rejected(self):
        model = FlowOCT3D(4)
        with self.assertRaises(ValueError):
            objectives(model, self.source, self.target, torch.tensor([.2, .3]), LossConfig())
        with self.assertRaises(ValueError):
            layer_masks(self.layers.flip(1), 32)
        with self.assertRaises(ValueError):
            LossConfig(rdca_slices=1)

    def test_baseline_and_full_share_initialization(self):
        states = []
        for name in ('02_haar_fm', '03_full', '04_no_rdca'):
            seed_all(20261021)
            model = FlowOCT3D(4, representation=arms()[name]['representation'])
            states.append(model.state_dict())
        for state in states[1:]:
            for k in state:
                torch.testing.assert_close(state[k], states[0][k], rtol=0., atol=0.)

    def test_shared_schedule_and_half_start(self):
        i, t = schedule(42, 800, 8)
        j, u = schedule(42, 800, 8, .5)
        np.testing.assert_array_equal(i, j)
        self.assertEqual(int((u == 0).sum()), 400)
        np.testing.assert_array_equal(t[u != 0], u[u != 0])
        np.testing.assert_array_equal(np.bincount(i), np.full(8, 100))

    def test_oct_only_rollout_sliding_oracle(self):
        class Constant(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(.03))
            def forward(self, state, time, source):
                return torch.ones_like(state)*self.bias
        c = torch.rand(1, 1, 19, 23, 21)*.5
        out = generate_volume(Constant(), c, (16, 16, 16), 8, .5, representation='voxel')
        torch.testing.assert_close(out, c+.03, atol=1e-6, rtol=1e-5)
        model = FlowOCT3D(4)
        self.assertEqual(model.rollout(self.source[:1], 2).shape, self.source[:1].shape)

    def test_checkpoint_resume_exact(self):
        model = FlowOCT3D(4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        config = LossConfig()
        def update(m, o, s, t):
            o.zero_grad(set_to_none=True)
            loss, _ = objectives(m, self.source, self.target, torch.tensor(t), config, self.layers)
            self.assertFalse(apply_update(loss, m, o, s)[0])
        update(model, optimizer, scaler, [.2, .3])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resume.pt'
            save_checkpoint(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict()))
            update(model, optimizer, scaler, [.6, .9])
            expected = copy.deepcopy(model.state_dict())
            restored = FlowOCT3D(4)
            opt = torch.optim.AdamW(restored.parameters(), lr=2e-4, weight_decay=0.)
            saved = torch.load(path, weights_only=False)
            restored.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
            update(restored, opt, scaler, [.6, .9])
            for k, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[k], rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main()

