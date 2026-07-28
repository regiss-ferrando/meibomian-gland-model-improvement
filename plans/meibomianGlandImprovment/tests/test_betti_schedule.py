import unittest

from train import betti_weight_for_epoch


class BettiScheduleTests(unittest.TestCase):
    def test_default_schedule_keeps_constant_weight(self):
        self.assertEqual(betti_weight_for_epoch(0.005, 0, 0, 1), 0.005)
        self.assertEqual(betti_weight_for_epoch(0.005, 0, 0, 50), 0.005)

    def test_warmup_and_linear_ramp(self):
        self.assertEqual(betti_weight_for_epoch(0.005, 10, 10, 1), 0.0)
        self.assertEqual(betti_weight_for_epoch(0.005, 10, 10, 10), 0.0)
        self.assertAlmostEqual(betti_weight_for_epoch(0.005, 10, 10, 11), 0.0005)
        self.assertAlmostEqual(betti_weight_for_epoch(0.005, 10, 10, 15), 0.0025)
        self.assertAlmostEqual(betti_weight_for_epoch(0.005, 10, 10, 20), 0.005)
        self.assertEqual(betti_weight_for_epoch(0.005, 10, 10, 21), 0.005)

    def test_zero_target_stays_disabled(self):
        self.assertEqual(betti_weight_for_epoch(0.0, 10, 10, 30), 0.0)

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            betti_weight_for_epoch(-0.1, 0, 0, 1)
        with self.assertRaises(ValueError):
            betti_weight_for_epoch(0.005, -1, 0, 1)
        with self.assertRaises(ValueError):
            betti_weight_for_epoch(0.005, 0, -1, 1)
        with self.assertRaises(ValueError):
            betti_weight_for_epoch(0.005, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
