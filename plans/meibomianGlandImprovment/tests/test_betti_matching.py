"""Tests for optional H0 Betti matching integration."""

import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from src.losses import BettiMatchingLossH0, CombinedLoss
from src.metrics import SegmentationMetrics


class _SyntheticMatchingResult:
    def __init__(self):
        empty = np.empty((0, 2), dtype=np.int64)
        self.input1_matched_birth_coordinates = [np.array([[1, 1]]), empty]
        self.input1_matched_death_coordinates = [np.array([[1, 2]]), empty]
        self.input2_matched_birth_coordinates = [np.array([[1, 1]]), empty]
        self.input2_matched_death_coordinates = [np.array([[1, 2]]), empty]
        self.input1_unmatched_birth_coordinates = [np.array([[2, 1]]), empty]
        self.input1_unmatched_death_coordinates = [np.array([[2, 2]]), empty]


class BettiMatchingTests(unittest.TestCase):
    def test_zero_weight_keeps_dependency_optional(self):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            criterion = CombinedLoss(betti_weight=0.0)
        logits = torch.zeros((1, 2, 4, 4), requires_grad=True)
        targets = torch.zeros((1, 4, 4), dtype=torch.long)
        loss = criterion(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_zero_weight_is_numerically_identical_to_baseline(self):
        logits = torch.randn((2, 2, 8, 8))
        targets = torch.randint(0, 2, (2, 8, 8))
        baseline = CombinedLoss()(logits, targets)
        optional_disabled = CombinedLoss(betti_weight=0.0)(logits, targets)
        self.assertTrue(torch.equal(baseline, optional_disabled))

    def test_positive_weight_requires_official_module(self):
        with mock.patch("importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(ImportError, "official betti_matching module"):
                CombinedLoss(betti_weight=0.01)

    def test_h0_loss_backpropagates_using_official_coordinates(self):
        fake_module = types.SimpleNamespace(
            compute_matching=lambda *args, **kwargs: [_SyntheticMatchingResult()]
        )
        with mock.patch.dict(sys.modules, {"betti_matching": fake_module}):
            criterion = BettiMatchingLossH0()
        logits = torch.randn((1, 2, 4, 4), requires_grad=True)
        targets = torch.zeros((1, 4, 4), dtype=torch.long)
        loss = criterion(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_betti0_metrics_detect_missing_and_additional_components(self):
        targets = torch.zeros((2, 8, 8), dtype=torch.long)
        targets[:, 1:3, 1:3] = 1
        targets[:, 5:7, 5:7] = 1
        predictions = targets.clone()
        predictions[0, 5:7, 5:7] = 0
        predictions[1, 1:3, 5:7] = 1

        metrics = SegmentationMetrics.betti0_metrics(predictions, targets)

        self.assertEqual(metrics["betti0_abs_error"], 1.0)
        self.assertEqual(metrics["betti0_missing_components"], 0.5)
        self.assertEqual(metrics["betti0_additional_components"], 0.5)


if __name__ == "__main__":
    unittest.main()
