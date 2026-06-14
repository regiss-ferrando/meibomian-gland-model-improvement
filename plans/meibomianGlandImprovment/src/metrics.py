"""
Segmentation evaluation metrics
"""

import torch
import torch.nn.functional as F
import numpy as np


class SegmentationMetrics:

    @staticmethod
    def _get_predictions(logits: torch.Tensor):
        """
        Convert logits -> predicted labels
        """

        if logits.dim() == 4:
            return torch.argmax(logits, dim=1)

        return logits

    @staticmethod
    def soft_dice(
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-5
    ) -> float:
        """
        Dice computed on probabilities.

        Useful during training.
        """

        probs = F.softmax(logits, dim=1)

        probs = probs[:, 1]

        targets = targets.float()

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + smooth
        ) / (
            probs.sum()
            + targets.sum()
            + smooth
        )

        return dice.item()

    @staticmethod
    def dice_coefficient(
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-5
    ) -> float:
        """
        Hard Dice score.
        """

        preds = SegmentationMetrics._get_predictions(
            logits
        )

        preds = preds.float().reshape(-1)

        targets = targets.float().reshape(-1)

        intersection = (
            preds * targets
        ).sum()

        dice = (
            2.0 * intersection + smooth
        ) / (
            preds.sum()
            + targets.sum()
            + smooth
        )

        return dice.item()

    @staticmethod
    def iou(
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-5
    ) -> float:
        """
        Intersection over Union.
        """

        preds = SegmentationMetrics._get_predictions(
            logits
        )

        preds = preds.float().reshape(-1)

        targets = targets.float().reshape(-1)

        intersection = (
            preds * targets
        ).sum()

        union = (
            preds.sum()
            + targets.sum()
            - intersection
        )

        iou = (
            intersection + smooth
        ) / (
            union + smooth
        )

        return iou.item()

    @staticmethod
    def pixel_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Pixel-wise accuracy.
        """

        preds = SegmentationMetrics._get_predictions(
            logits
        )

        correct = (
            preds == targets
        ).sum().item()

        total = targets.numel()

        return correct / total

    @staticmethod
    def mean_iou(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 2,
        smooth: float = 1e-5
    ) -> float:
        """
        Mean IoU across classes.
        """

        preds = SegmentationMetrics._get_predictions(
            logits
        )

        ious = []

        for cls in range(num_classes):

            pred_cls = (
                preds == cls
            ).float()

            target_cls = (
                targets == cls
            ).float()

            intersection = (
                pred_cls * target_cls
            ).sum()

            union = (
                pred_cls.sum()
                + target_cls.sum()
                - intersection
            )

            if union == 0:

                iou = torch.tensor(
                    1.0,
                    device=targets.device
                )

            else:

                iou = (
                    intersection + smooth
                ) / (
                    union + smooth
                )

            ious.append(
                iou.item()
            )

        return float(
            np.mean(ious)
        )

    @staticmethod
    def calculate_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 2
    ):
        """
        Compute all metrics.
        """
        preds = SegmentationMetrics._get_predictions(logits)

        return {
            "soft_dice":
                SegmentationMetrics.soft_dice(
                    logits,
                    targets
                ),

            "dice":
                SegmentationMetrics.dice_coefficient(
                    logits,
                    targets
                ),

            "iou":
                SegmentationMetrics.iou(
                    logits,
                    targets
                ),

            "pixel_accuracy":
                SegmentationMetrics.pixel_accuracy(
                    logits,
                    targets
                ),

            "mean_iou":
                SegmentationMetrics.mean_iou(
                    logits,
                    targets,
                    num_classes=num_classes
                ),

            "pred_foreground":
                (preds == 1).float().mean().item(),

            "target_foreground":
                (targets == 1).float().mean().item(),
        }
