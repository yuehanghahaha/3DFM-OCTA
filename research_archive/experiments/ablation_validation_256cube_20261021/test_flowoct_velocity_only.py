"""Check disabled supervision, original velocity gradients, and source-only solvers."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from flowoct_velocity_only import FlowOCTVelocityOnly, FlowOCT


def options(chunk=0):
    return SimpleNamespace(gpu_ids=[], isTrain=True, checkpoints_dir='.', name='test',
                           lr=2e-4, beta=.5, train_slices=0, network_chunk=chunk)


class VelocityOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_haar_initialization_and_velocity_gradients_match_original(self):
        torch.manual_seed(20261021)
        original = FlowOCT(options())
        torch.manual_seed(20261021)
        reduced = FlowOCTVelocityOnly(options(1), 'haar2d')
        for name, value in original.netG.state_dict().items():
            torch.testing.assert_close(value, reduced.netG.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(hasattr(reduced, 'contrast_projection_head'))
        original.L1weight = original.contrastweight = original.projweight = 0.
        batch = {'A': torch.rand(1, 1, 8, 32, 32), 'B': torch.rand(1, 1, 8, 32, 32)}
        # Warm the zero-initialized output, then compare nontrivial gradients.
        for index in range(2):
            original.set_input(batch)
            reduced.set_input(batch)
            torch.manual_seed(91 + index)
            original.forward()
            with patch.object(reduced, '_cropped_l1_loss', side_effect=AssertionError('L1 called')), \
                 patch.object(reduced, '_contrastive_loss', side_effect=AssertionError('contrast called')):
                torch.manual_seed(91 + index)
                reduced.forward()
            torch.testing.assert_close(original.loss, reduced.loss, atol=3e-6, rtol=1e-5)
            for name, parameter in original.netG.named_parameters():
                other = dict(reduced.netG.named_parameters())[name]
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=1e-3)

    def test_pixel_update_has_one_channel_and_no_wavelet_or_auxiliary_head(self):
        model = FlowOCTVelocityOnly(options(1), 'pixel')
        self.assertFalse(hasattr(model, 'dwt_2d'))
        self.assertFalse(hasattr(model, 'contrast_projection_head'))
        shapes = []
        hook = model.netG.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
        batch = {'A': torch.rand(1, 1, 8, 32, 32), 'B': torch.rand(1, 1, 8, 32, 32)}
        before = copy.deepcopy(model.netG.state_dict())
        model.set_input(batch)
        model.forward()
        hook.remove()
        self.assertTrue(all(s[1:] == (1, 32, 32) for s in shapes))
        self.assertEqual(len(model.last_sampling['slice_indices']), 2)
        self.assertEqual(len(model.last_sampling['slice_times']), 2)
        self.assertAlmostEqual(float(model.loss), 10 * float(model.vloss), places=6)
        self.assertTrue(any(not torch.equal(value, model.netG.state_dict()[key]) for key, value in before.items()))
        self.assertEqual({id(p) for group in model.optimizer_G.param_groups for p in group['params']},
                         {id(p) for p in model.netG.parameters()})

    def test_analytic_velocity_reaches_target_with_both_representations_and_solvers(self):
        class KnownVelocity(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer('values', values)

            def forward(self, state, times):
                return self.values.expand(len(state), -1, -1, -1)

        source = torch.full((2, 1, 32, 32), .2)
        target = torch.linspace(.3, .7, 32 * 32).reshape(1, 1, 32, 32).expand_as(source)
        for representation in ('pixel', 'haar2d'):
            model = FlowOCTVelocityOnly(options(), representation)
            velocity = (target - source)[:1]
            encoded = torch.cat(model.dwt_2d(velocity), dim=1) if representation == 'haar2d' else velocity
            model.netG = KnownVelocity(encoded)
            torch.testing.assert_close(model.generate(source), target, atol=2e-6, rtol=0)
            state = source
            for index in range(8):
                state = model.euler_step(state, torch.full((2,), index / 8),
                                         torch.full((2,), (index + 1) / 8))
            torch.testing.assert_close(state.clamp(0, 1), target, atol=2e-6, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
