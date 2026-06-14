"""
Loss functions for binary meibomian gland segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Cross-entropy plus foreground Dice loss."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_binary_segmentation_inputs(logits, targets)
        targets = targets.long()

        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.ce_weight * ce + self.dice_weight * dice


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
