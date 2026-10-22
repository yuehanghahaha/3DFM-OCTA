"""Z-axis, gradient, source-only supervision and checkpoint integration checks."""
import csv
import json
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
from .model import make_model, z_mean_projection, z_projection_loss


class RecordingVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.inputs = []

    def forward(self, state, time, source):
        self.inputs.append((state.detach().clone(), time.detach().clone(), source.detach().clone()))
        return torch.ones_like(state) * self.value


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projection_reduces_only_axial_z_and_uses_fp32(self):
        volume = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float16).reshape(2, 3, 4, 5, 6)
        volume.requires_grad_()
        actual = z_mean_projection(volume)
        self.assertEqual(actual.shape, (2, 3, 5, 6))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, sum(volume[:, :, z].float() for z in range(4)) / 4)
        actual.sum().backward()
        torch.testing.assert_close(volume.grad, torch.full_like(volume, 0.25))
        with self.assertRaises(ValueError):
            z_mean_projection(torch.zeros(2, 3, 4, 5))

    def test_supervision_uses_only_oct_at_zero_time_and_is_unclamped(self):
        source = torch.full((2, 1, 8, 16, 24), 0.1)
        target = torch.full_like(source, 0.2)
        for value, expected, gradient in ((2., 1.9, 1.), (-2., 2.1, -1.)):
            model = RecordingVelocity(value)
            loss = z_projection_loss(model, source, target)
            self.assertAlmostEqual(loss.item(), expected, places=5)
            loss.backward()
            self.assertAlmostEqual(model.value.grad.item(), gradient, places=5)
            state, time, condition = model.inputs[0]
            torch.testing.assert_close(state, source, rtol=0, atol=0)
            torch.testing.assert_close(condition, source, rtol=0, atol=0)
            torch.testing.assert_close(time, torch.zeros(2), rtol=0, atol=0)
        self.assertLess(z_projection_loss(RecordingVelocity(0.1), source, target).item(), 1e-6)

    def test_projection_gradient_reaches_the_unet(self):
        source = torch.rand(2, 1, 8, 16, 24)
        model = make_model(4)
        loss = z_projection_loss(model, source, source * 0.6)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.stem.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.output[-1].weight.grad.abs().sum().item(), 0)

    def test_validation_projection_metrics_handle_constant_maps(self):
        from .train import evaluate

        class Dataset:
            rows = [{'patient_id': 'a'}, {'patient_id': 'b'}]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                volume = torch.full((1, 8, 8, 8), 0.2)
                return {'oct': volume, 'octa': volume + 0.1, 'case_id': str(index)}

        report = evaluate(RecordingVelocity(0.1), Dataset(),
                          dict(patch=[8, 8, 8], ode_steps=2, overlap=0.5, amp=False, lambda_projection=0.1))
        self.assertLess(report['metrics']['z_projection_mae'], 1e-6)
        self.assertIsNone(report['metrics']['z_projection_pcc'])
        json.dumps(report, allow_nan=False)

    def test_train_resume_loss_configuration_and_evaluate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            rng = np.random.default_rng(22)
            for index in range(200):
                source = rng.random((8, 8, 8), dtype=np.float32)
                np.savez(root / f'{index}.npz', oct=source, octa=0.6 * source, axis_order=np.array('ZYX'))
                rows.append(dict(case_id=str(index), patient_id=str(index), path=f'{index}.npz'))
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

            common = ['-m', 'fm3d.train_projection', '--manifest', manifest, '--fov', '3M',
                      '--axis-order', 'ZYX', '--intensity-scale', '1', '--depth', '0',
                      '--patch', '8', '8', '8', '--base-channels', '4', '--steps-per-epoch', '2',
                      '--ode-steps', '2', '--val-every', '1', '--save-every', '1',
                      '--device', 'cpu', '--no-amp']
            run(*common, '--output', root / 'resumed', '--epochs', '1')
            run(*common, '--output', root / 'resumed', '--epochs', '2', '--resume')
            run(*common, '--output', root / 'continuous', '--epochs', '2')
            resumed = torch.load(root / 'resumed/latest.pt', weights_only=False)
            continuous = torch.load(root / 'continuous/latest.pt', weights_only=False)
            self.assertEqual(resumed['config']['lambda_projection'], 0.1)
            self.assertEqual(resumed['config']['representation'], 'voxel')
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key], continuous['model'][key], rtol=0, atol=0)
            history = [json.loads(line) for line in (root / 'resumed/history.jsonl').read_text().splitlines()]
            for record in history:
                self.assertAlmostEqual(record['loss'], record['fm_loss'] + 0.1 * record['z_projection_loss'], places=6)
                self.assertGreater(record['z_projection_loss'], 0)
                self.assertIn('z_projection_pcc', record['validation'])
                self.assertTrue((root / f'resumed/epoch_{record["epoch"]:04d}.pt').is_file())
            run(*common, '--output', root / 'resumed', '--epochs', '3', '--resume',
                '--lambda-projection', '0.2', error='Resume configuration changed: lambda_projection')
            run(*common, '--output', root / 'bad', '--lambda-projection', 'nan',
                error='lambda-projection must be finite and nonnegative')
            run(*common, '--output', root / 'bad', '--representation', 'haar3d',
                error='Z projection supervision currently requires representation=voxel')
            run('-m', 'fm3d.evaluate', '--checkpoint', root / 'resumed/best.pt',
                '--split', 'test', '--output', root / 'predictions', '--device', 'cpu')
            self.assertEqual(len(list((root / 'predictions').glob('*.npz'))), 50)
            metrics = json.loads((root / 'predictions/metrics.json').read_text())
            self.assertIn('z_projection_mae', metrics['metrics'])
            # Old baseline checkpoints have no projection key; weight=0 must resume unchanged.
            run(*common, '--lambda-projection', '0', '--output', root / 'legacy', '--epochs', '1')
            legacy_path = root / 'legacy/latest.pt'
            legacy = torch.load(legacy_path, weights_only=False)
            del legacy['config']['lambda_projection']
            torch.save(legacy, legacy_path)
            run(*common, '--lambda-projection', '0', '--output', root / 'legacy', '--epochs', '2', '--resume')


if __name__ == '__main__':
    unittest.main()
