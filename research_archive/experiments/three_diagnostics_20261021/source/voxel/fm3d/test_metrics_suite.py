import unittest
import numpy as np
from .metrics_suite import quality, vessel_scores, binary_scores, aggregate


class MetricsTest(unittest.TestCase):
    def test_identity_2d_and_3d(self):
        for shape in [(16, 16), (8, 16, 16)]:
            a = np.random.default_rng(4).random(shape).astype(np.float32)
            q = quality(a, a)
            self.assertEqual(q['mae'], 0)
            self.assertEqual(q['ssim'], 1)
            self.assertEqual(q['psnr'], 120)

    def test_mask_weighting_and_density(self):
        a = np.zeros((32, 32), np.float32)
        a[:16, :16] = 1
        b = a * 0.5
        mask = a > 0
        r, _, _ = vessel_scores(a, b, mask)
        self.assertAlmostEqual(r['weighted_mae'], 0.125)
        self.assertEqual(r['vde'], 0)
        self.assertAlmostEqual(r['vdc'], 1)
        bg, _, _ = vessel_scores(a, b, np.zeros_like(mask))
        self.assertAlmostEqual(bg['weighted_mae'], 0.0125)
        self.assertAlmostEqual(bg['weighted_mse'], 0.000625)

    def test_broken_centerline(self):
        a = np.zeros((32, 32), bool)
        a[16, 2:30] = True
        b = a.copy()
        b[16, 12:20] = False
        self.assertEqual(binary_scores(a, a)['cldice'], 1)
        self.assertLess(binary_scores(b, a)['cldice'], 1)
        self.assertEqual(binary_scores(~a, a)['dice'], 0)

    def test_empty_and_undefined(self):
        a = np.zeros((32, 32), np.float32)
        r, _, _ = vessel_scores(a, a, a.astype(bool))
        self.assertIsNone(r['vdc'])
        self.assertEqual(r['pred_vs_annotation_cldice'], 1)
        self.assertEqual(aggregate([{'case_id': 'a', 'x': 1.}, {'case_id': 'b', 'x': 3.}])['x']['mean'], 2)


if __name__ == '__main__':
    unittest.main()
