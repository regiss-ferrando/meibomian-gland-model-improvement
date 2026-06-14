"""
DeepLabv3+ model for meibomian gland segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet101_Weights, resnet50, resnet101


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates):
        super().__init__()
        modules = [
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
        ]
        for rate in atrous_rates:
            modules.append(
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
            )
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            ConvBNReLU(len(modules) * out_channels, out_channels, kernel_size=1),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        return self.project(torch.cat(features, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ with a ResNet encoder, ASPP, and low-level feature decoder."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_stride: int = 16,
    ):
        super().__init__()

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            atrous_rates = [12, 24, 36]
        else:
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")

        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            encoder = resnet50(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            high_channels = 2048
            low_channels = 256
        elif backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            encoder = resnet101(
                weights=weights,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            high_channels = 2048
            low_channels = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        original_conv = encoder.conv1
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                encoder.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
        )
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.aspp = ASPP(high_channels, 256, atrous_rates)
        self.low_project = ConvBNReLU(low_channels, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            ConvBNReLU(304, 256, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        input_size = x.shape[-2:]

        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(
            x,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_level = self.low_project(low_level)
        x = self.decoder(torch.cat([x, low_level], dim=1))
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return {"out": x}


def create_model(
    num_classes: int = 2,
    backbone: str = "resnet50",
    pretrained: bool = True,
    output_stride: int = 16,
    device: str = "cuda",
) -> nn.Module:
    model = DeepLabV3Plus(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        output_stride=output_stride,
    )

    return model.to(device)
