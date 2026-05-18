"""
TSM + ResNet-18 for Track 1 (Closed World, trained from scratch).

Temporal Shift Module (Lin et al., 2019) inserts a zero-parameter shift operation
before the first conv of every BasicBlock. 1/8 of channels are shifted one step
forward in time, 1/8 backward; the rest are unchanged.  The 2D convolutions that
follow naturally fuse spatial and temporal information, matching 3D-CNN accuracy
at 2D-CNN cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TemporalShift(nn.Module):
    """Zero-parameter temporal shift applied before a conv layer.

    Shifts fold = C // shift_div channels one step forward (+1) and the same
    number of channels one step backward (-1) along the time axis.
    Boundary positions are zero-padded.
    """

    def __init__(self, n_segment: int, shift_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W) — merged by the caller before any backbone layer
        B_T, C, H, W = x.shape
        T = self.n_segment
        B = B_T // T
        x = x.view(B, T, C, H, W)

        fold = C // self.shift_div
        out = torch.zeros_like(x)
        # channels 0:fold — forward shift (t-1 → t)
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # channels fold:2*fold — backward shift (t+1 → t)
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]
        # remaining channels — no shift
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]

        return out.view(B_T, C, H, W)


class TSMResNet18(nn.Module):
    """
    ResNet-18 backbone with Temporal Shift Modules injected into every BasicBlock.

    Input:  (B, T, C, H, W)  — standard dataset format
    Output: (B, num_classes) logits

    T must equal num_frames set at construction time.
    Trained from scratch (pretrained=False) for the Closed World track.

    temporal_pool:
        "mean" — average over all frames
        "last" — use only the last frame's features.  Because TSM forward-shifts
                 1/8 of channels at every block, the last frame accumulates context
                 from all preceding frames.  For action *anticipation* (partial clips)
                 this gives a compact representation of the most recent observable
                 state while retaining full temporal history.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        temporal_pool: str = "last",
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self._insert_tsm(backbone, num_frames)
        feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(feature_dim, num_classes))
        self.num_frames = num_frames
        self.temporal_pool = temporal_pool

    def _insert_tsm(self, model: nn.Module, n_segment: int) -> None:
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                block.conv1 = nn.Sequential(TemporalShift(n_segment), block.conv1)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape
        frames = video_batch.reshape(B * T, C, H, W)
        feats = self.backbone(frames).view(B, T, -1)
        if self.temporal_pool == "last":
            pooled = feats[:, -1]
        else:
            pooled = feats.mean(dim=1)
        return self.classifier(pooled)
