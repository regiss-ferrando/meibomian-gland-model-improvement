"""Unit tests for topology comparison helpers."""

import unittest

import numpy as np

from compare_topology_models import binary_metrics, difference_panel


class TopologyComparisonTests(unittest.TestCase):
    def test_binary_metrics_count_components_and_overlap(self):
        target = np.zeros((8, 8), dtype=np.uint8)
        target[1:3, 1:3] = 1
        target[5:7, 5:7] = 1
        prediction = target.copy()
        prediction[5:7, 5:7] = 0

        metrics = binary_metrics(prediction, target)

        self.assertEqual(metrics["beta0"], 1)
        self.assertEqual(metrics["beta0_error"], 1)
        self.assertAlmostEqual(metrics["dice"], 2.0 / 3.0, places=5)

    def test_difference_panel_uses_cyan_for_added_and_magenta_for_removed(self):
        image = np.zeros((3, 3), dtype=np.float32)
        baseline = np.zeros((3, 3), dtype=np.uint8)
        comparison = np.zeros((3, 3), dtype=np.uint8)
        baseline[0, 0] = 1
        comparison[2, 2] = 1

        panel = difference_panel(image, baseline, comparison)

        np.testing.assert_array_equal(panel[0, 0], np.array([1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(panel[2, 2], np.array([0.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
