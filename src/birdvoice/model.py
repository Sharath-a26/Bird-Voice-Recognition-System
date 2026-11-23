from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.body(x) + self.skip(x))


class BirdCNN(nn.Module):
    """CNN species classifier with an auxiliary pitch regression head."""
    def __init__(self, classes: int, width: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 5, 2, 2, bias=False), nn.BatchNorm2d(width), nn.GELU(),
            ResidualBlock(width, width), ResidualBlock(width, width * 2, 2),
            ResidualBlock(width * 2, width * 2), ResidualBlock(width * 2, width * 4, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, classes))
        self.pitch_head = nn.Linear(width * 4, 1)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.features(mel)
        return self.classifier(embedding), self.pitch_head(embedding).squeeze(-1)
