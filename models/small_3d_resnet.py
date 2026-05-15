from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("name", "Small3DResNetRegressor")
    if name != "Small3DResNetRegressor":
        raise ValueError(f"Unsupported model: {name}")

    return Small3DResNetRegressor(
        in_channels=int(config.get("in_channels", 2)),
        base_channels=int(config.get("base_channels", 16)),
        stages=tuple(config.get("stages", [16, 32, 64, 128])),
        blocks_per_stage=tuple(config.get("blocks_per_stage", [1, 1, 1, 1])),
        norm=str(config.get("norm", "groupnorm")),
        groupnorm_groups=int(config.get("groupnorm_groups", 8)),
        activation=str(config.get("activation", "silu")),
        dropout=float(config.get("dropout", 0.20)),
        head_dropout=float(config.get("head_dropout", 0.10)),
        output_dim=int(config.get("output_dim", 1)),
    )


class Small3DResNetRegressor(nn.Module):
    """Small 3D ResNet regressor for 2-channel MRI volumes."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
        stages: tuple[int, ...] = (16, 32, 64, 128),
        blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1),
        norm: str = "groupnorm",
        groupnorm_groups: int = 8,
        activation: str = "silu",
        dropout: float = 0.20,
        head_dropout: float = 0.10,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if len(stages) != len(blocks_per_stage):
            raise ValueError("stages and blocks_per_stage must have the same length.")

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=5, stride=2, padding=2, bias=False),
            make_norm(norm, base_channels, groupnorm_groups),
            make_activation(activation),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        current_channels = base_channels
        stage_modules = []
        for stage_index, (stage_channels, block_count) in enumerate(zip(stages, blocks_per_stage)):
            stride = 1 if stage_index == 0 else 2
            blocks = [
                ResidualBlock3D(
                    in_channels=current_channels,
                    out_channels=stage_channels,
                    stride=stride,
                    norm=norm,
                    groupnorm_groups=groupnorm_groups,
                    activation=activation,
                )
            ]
            current_channels = stage_channels

            for _ in range(1, block_count):
                blocks.append(
                    ResidualBlock3D(
                        in_channels=current_channels,
                        out_channels=stage_channels,
                        stride=1,
                        norm=norm,
                        groupnorm_groups=groupnorm_groups,
                        activation=activation,
                    )
                )

            stage_modules.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stage_modules)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current_channels, 64),
            make_activation(activation),
            nn.Dropout(head_dropout),
            nn.Linear(64, output_dim),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "groupnorm",
        groupnorm_groups: int = 8,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.activation = make_activation(activation)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = make_norm(norm, out_channels, groupnorm_groups)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = make_norm(norm, out_channels, groupnorm_groups)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(norm, out_channels, groupnorm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        out = self.activation(out)
        return out


def make_norm(name: str, channels: int, groupnorm_groups: int) -> nn.Module:
    if name.lower() == "groupnorm":
        groups = min(groupnorm_groups, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if name.lower() == "batchnorm":
        return nn.BatchNorm3d(channels)
    if name.lower() == "instancenorm":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"Unsupported norm: {name}")


def make_activation(name: str) -> nn.Module:
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
