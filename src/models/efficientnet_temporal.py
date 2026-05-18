"""
EfficientNetV2-S frame encoder + multi-layer temporal transformer.

Open World Track (Track 2): uses pretrained ImageNet weights and a much
stronger per-frame backbone than ResNet18. Key design choices:
  - EfficientNetV2-S backbone: 21 M params, 1280-d features, strong accuracy/speed tradeoff.
  - Gradient checkpointing on backbone.features (recomputes activations during backward
    instead of storing them) — cuts activation memory ~4x at ~30% compute overhead.
  - Sinusoidal positional encoding (fixed, robust to variable T at inference).
  - Pre-norm transformer layers (more stable training than post-norm).
  - CLS token + mean-pool fusion (captures both global summary and temporal mean).
  - param_groups() for differential learning-rate: backbone gets 10x lower LR.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import models


def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding, shape (1, seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class EfficientNetTemporal(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        pretrained: bool = True,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        feature_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.frame_norm = nn.LayerNorm(feature_dim)
        self.frame_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token; registered as a buffer so it moves with .to(device)
        self.register_buffer("pos_embed", _sinusoidal_pe(num_frames + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.head_norm = nn.LayerNorm(embed_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Run EfficientNetV2-S on a (B*T, C, H, W) tensor → (B*T, 1280)."""
        if self.use_checkpoint and self.training:
            # Checkpoint the conv stages only; avgpool + classifier are cheap.
            x = checkpoint_sequential(
                self.backbone.features, segments=4, input=frames, use_reentrant=False
            )
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return self.backbone.classifier(x)  # Identity → no-op
        return self.backbone(frames)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W)  →  logits: (B, num_classes)"""
        B, T, C, H, W = video_batch.shape
        frames = video_batch.reshape(B * T, C, H, W)

        feats = self._encode_frames(frames)               # (B*T, 1280)
        feats = feats.view(B, T, -1)                      # (B, T, 1280)
        feats = self.frame_norm(feats)
        tokens = self.frame_proj(feats)                   # (B, T, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, embed_dim)
        seq = torch.cat([cls, tokens], dim=1)             # (B, T+1, embed_dim)
        seq = seq + self.pos_embed[:, : seq.size(1), :]

        enc = self.temporal_encoder(seq)                  # (B, T+1, embed_dim)
        cls_out = enc[:, 0, :]                            # (B, embed_dim)
        mean_out = enc[:, 1:, :].mean(dim=1)              # (B, embed_dim)

        fused = torch.cat([cls_out, mean_out], dim=1)     # (B, 2*embed_dim)
        fused = self.head_norm(fused)
        return self.head(fused)                           # (B, num_classes)

    def param_groups(self, base_lr: float, backbone_scale: float = 0.1) -> List[dict]:
        """Return param groups with differential LR for use with AdamW."""
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": base_lr * backbone_scale},
            {"params": head_params, "lr": base_lr},
        ]
