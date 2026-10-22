"""Checks for native-coordinate slab selection and display/metric separation."""
import unittest

import numpy as np

from .projection_report import display, project, scores


class ProjectionTests(unittest.TestCase):
    def test_native_layer_endpoints_and_orientation(self):
        volume = np.zeros((640, 2, 3), dtype=np.float32)
        layers = np.broadcast_to(np.array([101, 151, 201, 251, 301, 401])[:, None, None],
                                 (6, 2, 3)).copy()
        volume[100, 0, 1] = 0.7  # ILM boundary must be included.
        volume[200, 1, 2] = 0.9  # OPL boundary in both closed slabs.
        volume[400, 0, 0] = 0.6  # BM boundary included.
        volume[99, 0, 1] = 1.0  # Outside the inner slab.
        maps, empty = project(volume, layers)
        self.assertEqual(maps['ILM_OPL'].shape, (3, 2))
        self.assertAlmostEqual(float(maps['ILM_OPL'][1, 0]), 0.7, places=6)
        self.assertAlmostEqual(float(maps['ILM_OPL'][2, 1]), 0.9, places=6)
        self.assertAlmostEqual(float(maps['OPL_BM'][0, 0]), 0.6, places=6)
        self.assertEqual(sum(empty.values()), 0)

    def test_z256_centers_use_half_pixel_mapping(self):
        volume = np.zeros((256, 2, 3), dtype=np.float32)
        layers = np.broadcast_to(np.array([101, 151, 201, 251, 301, 401])[:, None, None],
                                 (6, 2, 3)).copy()
        # Native centers: 40->100.75, 79->198.25, 80->200.75.
        volume[40, 0, 0] = 0.5
        volume[79, 0, 0] = 0.7
        volume[80, 0, 0] = 1.0
        maps, _ = project(volume, layers)
        self.assertAlmostEqual(float(maps['ILM_OPL'][0, 0]), 0.7, places=6)
        self.assertEqual(float(maps['OPL_BM'][0, 0]), 1.0)

    def test_display_does_not_change_raw_scores(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4) / 32
        b = a * 0.5
        before = scores(a, b)
        display(a)
        display(b, (0, 1))
        self.assertEqual(before, scores(a, b))
        self.assertGreater(before['mae'], 0)
        black, _ = display(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(np.asarray(black).sum(), 0)


if __name__ == '__main__':
    unittest.main()
