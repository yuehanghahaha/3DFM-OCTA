"""Tiny synthetic end-to-end check of production train/resume/OCT-only inference."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat
import torch


class CLITests(unittest.TestCase):
    def test_train_resume_and_source_only_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, layers = root/'data', root/'layers'
            data.mkdir()
            layers.mkdir()
            rng = np.random.default_rng(13)
            source = (rng.random((16, 16, 16))*250).astype(np.float32)
            target = source*.3
            layer_array = np.broadcast_to(np.array([80, 160, 240, 320, 400, 560], dtype=np.float32)[:, None, None], (6, 16, 16))
            manifest = root/'split.csv'
            with manifest.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id', 'path', 'split'])
                writer.writeheader()
                for i in range(200):
                    phase = 'train' if i < 140 else 'validation' if i < 150 else 'test'
                    cid = f'case{i:03d}'
                    path = data/(cid+'.npz')
                    np.savez_compressed(path, oct=source+i*.001, octa=target, axis_order=np.array('ZYX'))
                    writer.writerow(dict(case_id=cid, patient_id=cid, path=str(path), split=phase))
                    if phase != 'test':
                        savemat(layers/(cid+'.mat'), {'Layer': layer_array})
            common = ['--manifest', str(manifest), '--fov', '3M', '--layers-dir', str(layers),
                      '--depth', '16', '--patch', '16', '16', '16', '--base-channels', '4',
                      '--steps-per-epoch', '1', '--val-every', '1', '--ode-steps', '2', '--device', 'cpu', '--no-amp']
            def run(module, *args):
                result = subprocess.run([sys.executable, '-m', module, *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            run('flowoct3d.train', *common, '--epochs', '2', '--output', str(root/'continuous'))
            run('flowoct3d.train', *common, '--epochs', '1', '--output', str(root/'resumed'))
            run('flowoct3d.train', *common, '--epochs', '2', '--output', str(root/'resumed'), '--resume')
            a = torch.load(root/'continuous/latest.pt', map_location='cpu', weights_only=False)
            b = torch.load(root/'resumed/latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(a['updates'], 2)
            self.assertEqual(a['updates'], b['updates'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0., rtol=0.)
            for key in a['optimizer']['state']:
                for term in a['optimizer']['state'][key]:
                    torch.testing.assert_close(a['optimizer']['state'][key][term], b['optimizer']['state'][key][term], atol=0., rtol=0.)
            np.savez_compressed(root/'oct_only.npz', oct=source, axis_order=np.array('ZYX'))
            run('flowoct3d.predict', '--checkpoint', str(root/'resumed/latest.pt'), '--input', str(root/'oct_only.npz'),
                '--output', str(root/'prediction'), '--device', 'cpu', '--depth', '16', '--patch', '16', '16', '16',
                '--ode-steps', '2', '--no-amp')
            with np.load(root/'prediction/prediction.npz') as output:
                self.assertEqual(output['octa'].shape, (16, 16, 16))
                self.assertTrue(np.isfinite(output['octa']).all())
            run('flowoct3d.predict', '--checkpoint', str(root/'resumed/latest.pt'), '--manifest', str(manifest),
                '--fov', '3M', '--layers-dir', str(layers), '--split', 'validation', '--output', str(root/'evaluation'),
                '--device', 'cpu', '--depth', '16', '--patch', '16', '16', '16', '--ode-steps', '2', '--no-amp')
            report = json.loads((root/'evaluation/prediction.json').read_text())
            self.assertEqual(len(report['cases']), 10)
            self.assertEqual(report['summary']['validation']['patients'], 10)


if __name__ == '__main__':
    unittest.main()

