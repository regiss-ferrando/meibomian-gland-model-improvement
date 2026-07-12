"""
Loss functions for binary meibomian gland segmentation.
"""

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _validate_binary_segmentation_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape (B, C, H, W), got {tuple(logits.shape)}")
    if logits.size(1) != 2:
        raise ValueError(f"Expected 2 output classes for binary segmentation, got {logits.size(1)}")
    if targets.ndim != 3:
        raise ValueError(f"Expected targets with shape (B, H, W), got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
        raise ValueError(
            "Logits and targets spatial dimensions do not match: "
            f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
        )


class DiceLoss(nn.Module):
    """Soft Dice loss for the foreground class in binary segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)

        probs = F.softmax(logits, dim=1)[:, 1]
        targets = targets.float()

        intersection = (probs * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice_score


class CombinedLoss(nn.Module):
    """Cross-entropy plus Dice, with optional hard-negative and centerline losses."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        foreground_weight: Optional[float] = None,
        negative_weight: float = 0.0,
        hard_negative_percent: float = 0.1,
        hard_negative_min_prob: float = 0.0,
        cldice_weight: float = 0.0,
        cldice_iterations: int = 10,
        betti_weight: float = 0.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if foreground_weight is None:
            ce_class_weights = None
        else:
            ce_class_weights = torch.tensor([1.0, foreground_weight], dtype=torch.float32)
        self.register_buffer("ce_class_weights", ce_class_weights)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.negative_weight = negative_weight
        self.negative_loss = HardNegativeLoss(
            hard_negative_percent=hard_negative_percent,
            min_foreground_prob=hard_negative_min_prob,
        )
        self.cldice_weight = cldice_weight
        self.cldice_loss = SoftClDiceLoss(iterations=cldice_iterations, smooth=smooth)
        self.betti_weight = betti_weight
        self.betti_loss = BettiMatchingLossH0() if betti_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)
        targets = targets.long()

        ce = F.cross_entropy(logits, targets, weight=self.ce_class_weights)
        dice = self.dice_loss(logits, targets)
        loss = self.ce_weight * ce + self.dice_weight * dice

        if self.negative_weight > 0:
            loss = loss + self.negative_weight * self.negative_loss(logits, targets)

        if self.cldice_weight > 0:
            loss = loss + self.cldice_weight * self.cldice_loss(logits, targets)

        if self.betti_weight > 0:
            loss = loss + self.betti_weight * self.betti_loss(logits, targets)

        return loss


class BettiMatchingLossH0(nn.Module):
    """Betti matching loss restricted to connected components (H0).

    The official ``betti_matching`` C++ module identifies persistence-pair
    coordinates on detached CPU arrays. Loss values are then gathered from the
    original PyTorch probability tensor, so gradients flow to the network.
    """

    def __init__(self):
        super().__init__()
        try:
            self.betti_matching = importlib.import_module("betti_matching")
        except ImportError as exc:
            raise ImportError(
                "BettiMatchingLossH0 requires the official betti_matching module. "
                "Build nstucki/Betti-Matching-3D and add its build directory "
                "to PYTHONPATH. The baseline does not require this dependency "
                "when --betti-weight is zero."
            ) from exc

    @staticmethod
    def _coordinates(array, device: torch.device, dimensions: int) -> torch.Tensor:
        coordinates = np.asarray(array, dtype=np.int64)
        if coordinates.size == 0:
            return torch.empty((0, dimensions), dtype=torch.long, device=device)
        return torch.as_tensor(coordinates, dtype=torch.long, device=device)

    @staticmethod
    def _values_at(array: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[0] == 0:
            return array.new_empty((0,))
        return array[tuple(coordinates[:, dimension] for dimension in range(coordinates.shape[1]))]

    def _loss_for_image(self, prediction: torch.Tensor, target: torch.Tensor, result) -> torch.Tensor:
        dimensions = prediction.ndim
        device = prediction.device

        pred_match_birth = self._coordinates(
            result.input1_matched_birth_coordinates[0], device, dimensions
        )
        pred_match_death = self._coordinates(
            result.input1_matched_death_coordinates[0], device, dimensions
        )
        target_match_birth = self._coordinates(
            result.input2_matched_birth_coordinates[0], device, dimensions
        )
        target_match_death = self._coordinates(
            result.input2_matched_death_coordinates[0], device, dimensions
        )
        pred_unmatched_birth = self._coordinates(
            result.input1_unmatched_birth_coordinates[0], device, dimensions
        )
        pred_unmatched_death = self._coordinates(
            result.input1_unmatched_death_coordinates[0], device, dimensions
        )

        pred_matched = torch.stack(
            (
                self._values_at(prediction, pred_match_birth),
                self._values_at(prediction, pred_match_death),
            ),
            dim=1,
        )
        target_matched = torch.stack(
            (
                self._values_at(target, target_match_birth),
                self._values_at(target, target_match_death),
            ),
            dim=1,
        )
        pred_unmatched = torch.stack(
            (
                self._values_at(prediction, pred_unmatched_birth),
                self._values_at(prediction, pred_unmatched_death),
            ),
            dim=1,
        )

        matched_loss = 2.0 * ((pred_matched - target_matched) ** 2).sum()
        unmatched_loss = ((pred_unmatched[:, 0] - pred_unmatched[:, 1]) ** 2).sum()
        return matched_loss + unmatched_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)
        foreground = F.softmax(logits, dim=1)[:, 1]
        targets = targets.float()

        # The official implementation uses sublevel filtrations. Inverting the
        # foreground maps produces the desired foreground superlevel filtration.
        filtration_predictions = 1.0 - foreground
        filtration_targets = 1.0 - targets
        prediction_arrays = [
            np.ascontiguousarray(image.detach().cpu().numpy(), dtype=np.float64)
            for image in filtration_predictions
        ]
        target_arrays = [
            np.ascontiguousarray(image.detach().cpu().numpy(), dtype=np.float64)
            for image in filtration_targets
        ]
        results = self.betti_matching.compute_matching(
            prediction_arrays,
            target_arrays,
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=False,
        )
        losses = [
            self._loss_for_image(prediction, target, result)
            for prediction, target, result in zip(
                filtration_predictions, filtration_targets, results
            )
        ]
        return torch.stack(losses).mean()


class HardNegativeLoss(nn.Module):
    """Penalize confident foreground predictions on background pixels.

    This is useful when the model creates persistent false-positive gland blobs.
    It keeps ordinary supervised learning intact, then adds a targeted penalty on
    the background pixels where foreground probability is highest.
    """

    def __init__(
        self,
        hard_negative_percent: float = 0.1,
        min_foreground_prob: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hard_negative_percent <= 0 or hard_negative_percent > 1:
            raise ValueError(
                "hard_negative_percent must be in (0, 1], "
                f"got {hard_negative_percent}"
            )
        if min_foreground_prob < 0 or min_foreground_prob >= 1:
            raise ValueError(
                "min_foreground_prob must be in [0, 1), "
                f"got {min_foreground_prob}"
            )
        self.hard_negative_percent = hard_negative_percent
        self.min_foreground_prob = min_foreground_prob
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)

        foreground_probs = F.softmax(logits, dim=1)[:, 1]
        background_probs = foreground_probs[targets.long() == 0]
        if background_probs.numel() == 0:
            return logits.new_tensor(0.0)

        if self.min_foreground_prob > 0:
            background_probs = background_probs[background_probs >= self.min_foreground_prob]
            if background_probs.numel() == 0:
                return logits.new_tensor(0.0)

        n_hard = max(1, int(background_probs.numel() * self.hard_negative_percent))
        hard_probs = torch.topk(background_probs, k=n_hard, largest=True).values
        hard_probs = hard_probs.clamp(min=self.eps, max=1.0 - self.eps)

        return -torch.log1p(-hard_probs).mean()


class SoftClDiceLoss(nn.Module):
    """Soft centerline Dice loss for thin, tubular segmentation targets.

    The loss compares differentiable skeletons of the predicted foreground and
    target mask. It encourages thin structures to stay connected and discourages
    turning long glands into thick blobs.
    """

    def __init__(self, iterations: int = 10, smooth: float = 1e-5):
        super().__init__()
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.iterations = iterations
        self.smooth = smooth

    @staticmethod
    def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
        vertical = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
        horizontal = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(vertical, horizontal)

    @staticmethod
    def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    def _soft_open(self, mask: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(mask))

    def _soft_skeletonize(self, mask: torch.Tensor) -> torch.Tensor:
        opened = self._soft_open(mask)
        skeleton = F.relu(mask - opened)

        eroded = mask
        for _ in range(self.iterations):
            eroded = self._soft_erode(eroded)
            opened = self._soft_open(eroded)
            delta = F.relu(eroded - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)

        probs = F.softmax(logits, dim=1)[:, 1:2]
        targets = targets.float().unsqueeze(1)

        pred_skeleton = self._soft_skeletonize(probs)
        target_skeleton = self._soft_skeletonize(targets)

        topology_precision = (pred_skeleton * targets).sum() / (
            pred_skeleton.sum() + self.smooth
        )
        topology_sensitivity = (target_skeleton * probs).sum() / (
            target_skeleton.sum() + self.smooth
        )
        cldice = (2.0 * topology_precision * topology_sensitivity + self.smooth) / (
            topology_precision + topology_sensitivity + self.smooth
        )

        return 1.0 - cldice


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)

        ce_loss = F.cross_entropy(logits, targets.long(), reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal.mean()


class FocalDiceLoss(nn.Module):
    """Focal loss plus foreground Dice loss."""

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
