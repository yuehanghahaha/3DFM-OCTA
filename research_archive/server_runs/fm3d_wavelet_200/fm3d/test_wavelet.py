"""Mathematical and training/inference checks for the 3D Haar experiment."""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import make_splits
from .inference import generate_volume
from .model import flow_loss, make_model
from .wavelets import SUBBANDS, haar_dwt3d, haar_idwt3d, validate_wavelet_patch


class SourceVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, state, time, source):
        return self.value * source


class ConstantVoxelVelocity(SourceVelocity):
    def forward(self, state, time, source):
        # A spatially constant voxel field has only LLL coefficients.
        result = torch.zeros_like(state)
        result[:, 0] = self.value * math.sqrt(8)
        return result


class ExcursionVelocity(ConstantVoxelVelocity):
    def forward(self, state, time, source):
        sign = torch.where(time < 0.5, -1., 1.)[:, None, None, None, None]
        return sign * super().forward(state, time, source)


class WaveletTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inverse_energy_and_gradient(self):
        volume = torch.randn(2, 3, 16, 18, 20, dtype=torch.float64, requires_grad=True)
        coefficients = haar_dwt3d(volume)
        self.assertEqual(coefficients.shape, (2, 24, 8, 9, 10))
        torch.testing.assert_close(haar_idwt3d(coefficients), volume, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(coefficients.square().sum(), volume.square().sum())
        gradient, = torch.autograd.grad(coefficients.square().sum(), volume)
        torch.testing.assert_close(gradient, 2 * volume, atol=1e-12, rtol=1e-12)
        small = torch.randn(1, 1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: haar_idwt3d(haar_dwt3d(x)), (small,)))

    def test_all_eight_band_axes_and_signs(self):
        for band_index, band in enumerate(SUBBANDS):
            axes = [torch.tensor([1., 1.] if symbol == 'L' else [1., -1.]) for symbol in band]
            volume = (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])[None, None]
            actual = haar_dwt3d(volume).flatten()
            expected = torch.zeros(8)
            expected[band_index] = math.sqrt(8)
            torch.testing.assert_close(actual, expected)

    def test_matches_pywavelets_if_installed(self):
        try:
            import pywt
        except ImportError:
            self.skipTest('Optional reference check; PyWavelets is not a runtime dependency')
        volume = torch.randn(2, 2, 6, 8, 10, dtype=torch.float64)
        actual = haar_dwt3d(volume)
        for batch in range(2):
            for channel in range(2):
                reference = pywt.dwtn(volume[batch, channel].numpy(), 'haar', axes=(0, 1, 2))
                for index, band in enumerate(SUBBANDS):
                    key = band.lower().replace('l', 'a').replace('h', 'd')
                    np.testing.assert_allclose(actual[batch, index * 2 + channel].numpy(),
                                               reference[key], atol=1e-12, rtol=1e-12)

    def test_flow_oracle_and_mse_scaling(self):
        source = torch.rand(2, 1, 16, 16, 16)
        target = 0.6 * source
        for time in (torch.zeros(2), torch.ones(2), torch.tensor([0.2, 0.7])):
            loss = flow_loss(SourceVelocity(-0.4), source, target, time, representation='haar3d')
            self.assertLess(loss.item(), 1e-12)
        # Equal weights and orthonormal scaling must preserve voxel-space MSE.
        loss = flow_loss(SourceVelocity(0.3), source, target, representation='haar3d')
        torch.testing.assert_close(loss, (0.3 * source - (target - source)).square().mean())

    def test_conditioned_and_unconditioned_gradients(self):
        source = torch.rand(2, 1, 16, 32, 16)
        for condition in (True, False):
            model = make_model(4, condition, 'haar3d')
            self.assertEqual(model.stem.in_channels, 16 if condition else 8)
            self.assertEqual(model.output[-1].out_channels, 8)
            original = model.stem.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = flow_loss(model, source, source * 0.7, representation='haar3d')
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
            self.assertFalse(torch.equal(original, model.stem.weight))

    def test_global_coefficient_stitching_padding_and_fixed_condition(self):
        for shape in ((17, 23, 31), (8, 9, 10)):
            source = torch.rand(1, 1, *shape) * 0.5
            for model, expected in ((ConstantVoxelVelocity(0.2), source + 0.2),
                                    (SourceVelocity(0.2), source * 1.2)):
                prediction = generate_volume(model, source, (16, 16, 16), steps=3,
                                             overlap=0.6, representation='haar3d')
                torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)
                self.assertTrue(model.training)
        # Signed coefficients and intermediate states must not be clamped.
        source = torch.full((1, 1, 16, 16, 16), 0.2)
        prediction = generate_volume(ExcursionVelocity(0.8), source, (16, 16, 16),
                                     steps=2, representation='haar3d')
        torch.testing.assert_close(prediction, source, atol=1e-6, rtol=1e-6)

    def test_invalid_shapes_fail_explicitly(self):
        for patch in ((16, 16, 15), (8, 16, 16), (16, 16)):
            with self.assertRaises(ValueError):
                validate_wavelet_patch(patch)
        with self.assertRaises(ValueError):
            haar_dwt3d(torch.rand(1, 1, 17, 16, 16))
        with self.assertRaises(ValueError):
            haar_idwt3d(torch.rand(1, 7, 8, 8, 8))
        with self.assertRaises(ValueError):
            haar_dwt3d(torch.zeros(1, 1, 2, 2, 2, dtype=torch.uint8))

    def test_train_resume_milestones_and_evaluate_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(21)
            rows = []
            for index in range(200):
                source = rng.random((16, 16, 16), dtype=np.float32)
                np.savez(root / f'{index}.npz', oct=source, octa=0.6 * source, axis_order=np.array('ZYX'))
                rows.append({'case_id': str(index), 'patient_id': str(index), 'path': f'{index}.npz'})
            inventory = root / 'inventory.csv'
            with inventory.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id', 'path'])
                writer.writeheader()
                writer.writerows(rows)
            manifest = root / 'split.csv'
            make_splits(inventory, manifest, '3M', 2026)
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

            def run(*args, error=None):
                result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                                        text=True, env=env, timeout=120)
                output = result.stdout + result.stderr
                if error is None:
                    self.assertEqual(result.returncode, 0, output)
                else:
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(error, output)

            common = ['-m', 'fm3d.train_wavelet', '--manifest', manifest, '--fov', '3M',
                      '--axis-order', 'ZYX', '--intensity-scale', '1', '--depth', '0',
                      '--patch', '16', '16', '16', '--base-channels', '4', '--steps-per-epoch', '2',
                      '--ode-steps', '2', '--val-every', '1', '--save-every', '1',
                      '--device', 'cpu', '--no-amp']
            run(*common, '--output', root / 'resumed', '--epochs', '1')
            run(*common, '--output', root / 'resumed', '--epochs', '2', '--resume')
            run(*common, '--output', root / 'continuous', '--epochs', '2')
            resumed = torch.load(root / 'resumed/latest.pt', weights_only=False)
            continuous = torch.load(root / 'continuous/latest.pt', weights_only=False)
            self.assertEqual(resumed['epoch'], 2)
            self.assertEqual(resumed['config']['representation'], 'haar3d')
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key], continuous['model'][key], rtol=0, atol=0)
            for epoch in (1, 2):
                epoch_ckpt = torch.load(root / f'resumed/epoch_{epoch:04d}.pt', weights_only=False)
                best_ckpt = torch.load(root / f'resumed/best_through_{epoch:04d}.pt', weights_only=False)
                self.assertEqual(epoch_ckpt['epoch'], epoch)
                self.assertLessEqual(best_ckpt['epoch'], epoch)
                self.assertEqual(best_ckpt['best_mae'], epoch_ckpt['best_mae'])
            run(*common, '--output', root / 'resumed', '--epochs', '3', '--resume',
                '--representation', 'voxel', error='Resume configuration changed: representation')
            run('-m', 'fm3d.evaluate', '--checkpoint', root / 'resumed/best.pt',
                '--split', 'test', '--output', root / 'predictions', '--device', 'cpu')
            self.assertEqual(len(list((root / 'predictions').glob('*.npz'))), 50)
            with np.load(next((root / 'predictions').glob('*.npz'))) as data:
                self.assertEqual(data['octa'].shape, (16, 16, 16))
                self.assertTrue(np.isfinite(data['octa']).all())
                self.assertEqual(str(data['axis_order']), 'ZYX')
            metrics = json.loads((root / 'predictions/metrics.json').read_text())
            self.assertEqual(metrics['config']['representation'], 'haar3d')
            self.assertEqual(metrics['patients'], 50)


if __name__ == '__main__':
    unittest.main()
