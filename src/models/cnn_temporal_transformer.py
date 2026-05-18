"""
CNN frame encoder + temporal transformer for closed-world video classification.

The dataset clips are short and contain only four frames per video. This model
keeps the provided ResNet18-style frame encoder, but replaces uniform temporal
pooling with a learnable transformer over the frame sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNTemporalTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.frame_norm = nn.LayerNorm(feature_dim)
        self.frame_projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_frames + 1, embed_dim)
        )
        self.temporal_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(embed_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        frame_features = self.frame_norm(frame_features)

        tokens = self.frame_projection(frame_features)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, tokens], dim=1)
        sequence = sequence + self.positional_embedding[:, : sequence.size(1), :]
        sequence = self.temporal_dropout(sequence)

        encoded = self.temporal_encoder(sequence)
        cls_representation = encoded[:, 0, :]
        pooled_representation = encoded[:, 1:, :].mean(dim=1)

        fused = torch.cat([cls_representation, pooled_representation], dim=1)
        fused = self.output_norm(fused)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return logits