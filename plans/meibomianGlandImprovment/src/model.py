"""
DeepLabv3+ model for meibomian gland segmentation
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ model wrapper for meibomian gland segmentation"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 output_stride: int = 16):
        """
        Initialize DeepLabv3+ model
        
        Args:
            num_classes: Number of output classes (2 for binary segmentation)
            backbone: Backbone architecture ("resnet50" or "resnet101")
            pretrained: Whether to use pretrained weights
            output_stride: Output stride (16 or 8)
        """
        super().__init__()
        
        # Use ImageNet backbone weights only. Loading segmentation weights would
        # expect the original COCO/VOC class head instead of our 2-class head.
        if backbone == "resnet50":
            weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
            )
        elif backbone == "resnet101":
            weights_backbone = ResNet101_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet101(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify for single-channel input
        original_conv = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Copy pretrained weights if available (take average of RGB channels)
        if pretrained:
            with torch.no_grad():
                self.model.backbone.conv1.weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, H, W)
            
        Returns:
            Dictionary with output and auxiliary outputs
        """
        return self.model(x)


def create_model(num_classes: int = 2,
                backbone: str = "resnet50",
                pretrained: bool = True,
                device: str = "cuda") -> nn.Module:
    """
    Create and move model to device
    
    Args:
        num_classes: Number of classes
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        device: Device to move model to
        
    Returns:
        Model on specified device
    """
    model = DeepLabV3Plus(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    
    return model.to(device)
