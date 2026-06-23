"""
Loss functions for binary meibomian gland segmentation.
"""

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
    """Cross-entropy plus foreground Dice loss, with optional hard-negative learning."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        foreground_weight: Optional[float] = None,
        negative_weight: float = 0.0,
        hard_negative_percent: float = 0.1,
        hard_negative_min_prob: float = 0.0,
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)
        targets = targets.long()

        ce = F.cross_entropy(logits, targets, weight=self.ce_class_weights)
        dice = self.dice_loss(logits, targets)
        loss = self.ce_weight * ce + self.dice_weight * dice

        if self.negative_weight > 0:
            loss = loss + self.negative_weight * self.negative_loss(logits, targets)

        return loss


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
