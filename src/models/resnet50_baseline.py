from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Baseline(nn.Module):
    """ResNet50 (pretrained) + temporal mean pooling. Open World Track."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W) → logits: (B, num_classes)"""
        B, T, C, H, W = video_batch.shape
        frames = video_batch.reshape(B * T, C, H, W)
        feats = self.backbone(frames).view(B, T, -1).mean(dim=1)  # (B, 2048)
        return self.classifier(feats)

    def param_groups(self, base_lr: float, backbone_scale: float = 0.1) -> List[dict]:
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": base_lr * backbone_scale},
            {"params": head_params, "lr": base_lr},
        ]
