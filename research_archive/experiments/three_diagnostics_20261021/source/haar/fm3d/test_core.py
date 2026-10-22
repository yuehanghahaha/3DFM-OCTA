"""Run: python -m unittest fm3d.test_core -v. Uses synthetic data only."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .data import COUNTS, VolumeDataset, load_pair, make_splits, pack_bmp, read_rows, validate_manifest
from .inference import generate_volume
from .model import VelocityUNet3D, flow_loss


class ConstantVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, state, time, source):
        return torch.ones_like(state) * self.value


class ConditionVelocity(ConstantVelocity):
    def forward(self, state, time, source):
        return self.value * source


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flow_target_and_batched_gradient(self):
        a = torch.rand(2, 1, 8, 16, 8) * 0.5
        b = a + 0.25
        for t in (torch.zeros(2), torch.ones(2), torch.tensor([0.2, 0.7])):
            self.assertLess(flow_loss(ConstantVelocity(0.25), a, b, t).item(), 1e-12)
        for condition in (True, False):
            model = VelocityUNet3D(4, condition)
            before = model.stem.weight.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = flow_loss(model, a, b)
            loss.backward()
            self.assertTrue(torch.isfinite(model.stem.weight.grad).all())
            optimizer.step()
            self.assertFalse(torch.equal(before, model.stem.weight))

    def test_overlap_coverage_padding_and_fixed_source(self):
        for shape in ((8, 11, 15), (5, 7, 9)):
            a = torch.rand(1, 1, *shape) * 0.5
            model = ConstantVelocity(0.2)
            actual = generate_volume(model, a, (8, 8, 8), steps=3, overlap=0.5)
            torch.testing.assert_close(actual, a + 0.2, atol=1e-6, rtol=1e-6)
            self.assertTrue(model.training)
            actual = generate_volume(ConditionVelocity(0.2), a, (8, 8, 8), steps=4)
            torch.testing.assert_close(actual, a * 1.2, atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for AMP overflow test')
    def test_amp_overflow_skips_then_recovers(self):
        from .train import apply_update
        model = nn.Linear(2, 1).cuda()
        with torch.no_grad():
            model.weight.fill_(0.1)
            model.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler('cuda', init_scale=16)
        original = model.weight.detach().clone()
        hook = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        with torch.autocast('cuda'):
            loss = (model(torch.ones(1, 2, device='cuda')) - 2).float().square().mean()
        self.assertTrue(apply_update(loss, model, optimizer, scaler))
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
        self.assertEqual(scaler.get_scale(), 8)
        hook.remove()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss = (model(torch.ones(1, 2, device='cuda')) - 2).float().square().mean()
        self.assertFalse(apply_update(loss, model, optimizer, scaler))
        self.assertFalse(torch.equal(model.weight, original))

    def test_axes_scale_and_paired_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'case.npz'
            a = np.arange(12 * 16 * 8).reshape(12, 16, 8).astype('float32') % 100
            np.savez(path, oct=a, octa=a + 100, axis_order=np.array('YZX'))
            source, target = load_pair(path, 'YZX', 255, 0)
            self.assertEqual(source.shape, (1, 16, 12, 8))
            torch.testing.assert_close(source[0], torch.from_numpy(a.transpose(1, 0, 2).copy()) / 255)
            rows = [{'case_id': 'a', 'path': str(path), 'split': 'train'}]
            item = VolumeDataset(rows, 'train', 'YZX', 255, 8, (8, 8, 8))[0]
            torch.testing.assert_close(item['octa'] - item['oct'], torch.full((1, 8, 8, 8), 100 / 255))
            with self.assertRaises(ValueError):
                load_pair(path, 'ZYX', 255, 0)
            with self.assertRaises(ValueError):
                load_pair(path, 'YZX', 1, 0)

    def test_bmp_pack_is_lossless_and_numeric(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('oct', 'octa'):
                (root / name).mkdir()
                for i in range(1, 13):
                    Image.fromarray(np.full((16, 8), i, dtype='uint8')).save(root / name / f'{i}.bmp')
            output = root / 'pair.npz'
            pack_bmp(root / 'oct', root / 'octa', output, 'YZX')
            with np.load(output) as data:
                self.assertEqual(data['oct'].shape, (16, 12, 8))
                np.testing.assert_array_equal(data['oct'][0, :, 0], np.arange(1, 13))
                self.assertEqual(data['oct'].dtype, np.uint8)
            (root / 'octa' / '3.bmp').unlink()
            with self.assertRaises(ValueError):
                pack_bmp(root / 'oct', root / 'octa', root / 'bad.npz', 'YZX')

    def test_native_and_cached_geometry_have_identical_model_inputs(self):
        coordinates = (np.arange(256) + 0.5) * 640 / 256 - 0.5
        lower = coordinates.astype(int)
        weight = (coordinates - lower).astype(np.float32)[:, None, None]
        def cached_resize(array):
            return array[lower].astype(np.float32) * (1 - weight) + array[lower + 1].astype(np.float32) * weight
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = np.random.default_rng(71).integers(0, 256, (640, 19, 23), dtype=np.uint8)
            b = np.flip(a, axis=2).copy()
            np.savez(root / 'native.npz', oct=a, octa=b, axis_order=np.array('ZYX'))
            np.savez(root / 'cache.npz', oct=cached_resize(a), octa=cached_resize(b), axis_order=np.array('ZYX'))
            native = load_pair(root / 'native.npz', 'ZYX', 255, 256)
            cached = load_pair(root / 'cache.npz', 'ZYX', 255, 256)
            for x, y in zip(native, cached):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_manifest_counts_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for fov in ('3M', '6M'):
                rows = []
                for i in range(sum(COUNTS[fov].values())):
                    path = root / f'{i}.npz'
                    path.touch()
                    rows.append({'case_id': str(i), 'patient_id': str(i), 'path': path.name})
                inventory = root / f'{fov}_inventory.csv'
                with inventory.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id', 'path'])
                    writer.writeheader()
                    writer.writerows(rows)
                output = root / f'{fov}.csv'
                make_splits(inventory, output, fov, 2026)
                assigned = read_rows(output)
                validate_manifest(assigned, fov)
                with self.assertRaises(FileExistsError):
                    make_splits(inventory, output, fov, 2026)
                assigned[0]['leakage_group'] = assigned[-1]['leakage_group'] = 'same_pixels'
                with self.assertRaisesRegex(ValueError, 'Duplicate image content'):
                    validate_manifest(assigned, fov)
                assigned[-1]['leakage_group'] = 'different_pixels'
                validate_manifest(assigned, fov)
                assigned[-1]['patient_id'] = assigned[0]['patient_id']
                with self.assertRaises(ValueError):
                    validate_manifest(assigned, fov)
                with self.assertRaises(ValueError):
                    validate_manifest(assigned[:-1], fov)

    def test_duplicate_group_repair_preserves_counts(self):
        from collections import Counter
        from tools.prepare_common_octa500 import protect_duplicate_groups
        import copy
        rows = [dict(case_id=str(i), patient_id=str(i), fov='3M', disease='NORMAL',
                     sex='M', eye='OS', age=str(50 + i), split=phase)
                for i, phase in enumerate(['train', 'test', 'validation', 'train', 'train', 'test'])]
        records = [dict(case_id=str(i), source={
            'oct': {'array_sha256': str(0 if i == 1 else i)},
            'octa': {'array_sha256': str(1 if i == 2 else i)}}) for i in range(6)]
        original, second = Counter(r['split'] for r in rows), copy.deepcopy(rows)
        report = protect_duplicate_groups(rows, records)
        self.assertEqual(Counter(r['split'] for r in rows), original)
        self.assertEqual([r['split'] for r in rows[:3]], ['train'] * 3)
        self.assertEqual(len({r['leakage_group'] for r in rows[:3]}), 1)
        self.assertEqual(report['unique_content_groups'], 4)
        self.assertEqual(report, protect_duplicate_groups(second, records))
        self.assertEqual(rows, second)
        impossible = copy.deepcopy(second[:3])
        impossible[1]['split'] = 'test'
        with self.assertRaises(ValueError):
            protect_duplicate_groups(impossible, records[:3])

    def test_train_resume_and_evaluate_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(3)
            rows = []
            for i in range(200):
                source = rng.random((8, 8, 8), dtype=np.float32)
                np.savez(root / f'{i}.npz', oct=source, octa=0.5 * source, axis_order=np.array('ZYX'))
                rows.append({'case_id': str(i), 'patient_id': str(i), 'path': f'{i}.npz'})
            inventory = root / 'inventory.csv'
            with inventory.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id', 'path'])
                writer.writeheader()
                writer.writerows(rows)
            manifest = root / 'split.csv'
            make_splits(inventory, manifest, '3M', 2026)
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            def run(*args):
                result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                                        text=True, env=env, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            common = ['-m', 'fm3d.train', '--manifest', manifest, '--fov', '3M',
                      '--axis-order', 'ZYX', '--intensity-scale', '1', '--depth', '0',
                      '--patch', '8', '8', '8', '--base-channels', '4', '--steps-per-epoch', '2',
                      '--ode-steps', '2', '--val-every', '1', '--device', 'cpu', '--no-amp']
            run(*common, '--output', root / 'resumed', '--epochs', '1')
            run(*common, '--output', root / 'resumed', '--epochs', '2', '--resume')
            run(*common, '--output', root / 'continuous', '--epochs', '2')
            resumed = torch.load(root / 'resumed/latest.pt', weights_only=False)
            continuous = torch.load(root / 'continuous/latest.pt', weights_only=False)
            self.assertEqual(resumed['epoch'], 2)
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key], continuous['model'][key], rtol=0, atol=0)
            run('-m', 'fm3d.evaluate', '--checkpoint', root / 'resumed/best.pt',
                '--split', 'test', '--output', root / 'predictions', '--device', 'cpu')
            self.assertEqual(len(list((root / 'predictions').glob('*.npz'))), 50)


if __name__ == '__main__':
    unittest.main()
