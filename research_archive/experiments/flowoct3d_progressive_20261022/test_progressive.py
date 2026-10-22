import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from engine import from_public, to_public, public_prediction, public_probe, make_public
from flowoct3d.utils import read_json, seed_all


class ProgressiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cumulative_switches(self):
        arms = read_json(ROOT/'protocol.json')['arms']
        names = list(arms)[1:7]
        previous = arms[names[0]]
        expected = [('rdca_weight', 0.), ('projection_weight', 0.), ('structure', False), ('reconstruction_weight', 0.)]
        for name, (key, value) in zip(names[1:5], expected):
            current = arms[name]
            changed = {k for k in previous['losses'] if previous['losses'][k] != current['losses'][k]}
            self.assertEqual(changed, {key})
            self.assertEqual(current['losses'][key], value)
            previous = current
        self.assertEqual(arms[names[4]]['losses'], arms[names[5]]['losses'])
        self.assertEqual(arms[names[4]]['representation'], 'haar3d')
        self.assertEqual(arms[names[5]]['representation'], 'voxel')

    def test_schedule_prefix_balance(self):
        config = read_json(ROOT/'protocol.json')
        with np.load(ROOT/'data/schedule.npz') as s, np.load(Path(config['prior_study'])/'03_full/few_case/schedule.npz') as old:
            np.testing.assert_array_equal(s['indices'][:800], old['indices'])
            np.testing.assert_array_equal(s['times'][:800], old['times'])
            np.testing.assert_array_equal(np.bincount(s['indices']), np.full(8, 300))
            self.assertEqual(int((s['half_times'] == 0).sum()), 1200)
            np.testing.assert_array_equal(s['half_times'][s['half_times'] != 0], s['times'][s['half_times'] != 0])

    def test_physical_axis_roundtrip_and_pure_oct_solver(self):
        x = torch.arange(2*1*32*16*48).reshape(2, 1, 32, 16, 48)
        torch.testing.assert_close(from_public(to_public(x)), x, atol=0, rtol=0)
        torch.testing.assert_close(to_public(x)[1, 0, 5], x[1, 0, :, 5, :].T, atol=0, rtol=0)
        class Oracle:
            def euler_step(self, state, start, end):
                return state+(end-start)[:, None, None, None]*.1
        for steps, clamp in ((8, False), (3, True)):
            source = torch.full((1, 1, 32, 16, 32), .2)
            torch.testing.assert_close(public_prediction(Oracle(), source, steps, clamp, 3), source+.1, atol=1e-6, rtol=0)

    def test_public_update_and_probe(self):
        seed_all(42)
        model = make_public(ROOT, torch.device('cpu'))
        source = torch.rand(1, 1, 32, 16, 32)*.7
        target = source*.4
        model.set_input({'A': to_public(source), 'B': to_public(target)})
        before = copy.deepcopy(model.netG.state_dict())
        model.forward()
        parts = model.get_current_loss()
        self.assertAlmostEqual(float(model.loss), 10*parts['vloss']+.5*parts['L1loss']+.005*parts['contrastloss']+.1*parts['proj_loss'], places=5)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.netG.state_dict().items()))
        mse, endpoint = public_probe(model, source, target, .5, 4)
        self.assertGreaterEqual(mse, 0)
        self.assertEqual(endpoint.shape, source.shape)
        self.assertTrue(torch.isfinite(endpoint).all())


if __name__ == '__main__':
    unittest.main()

