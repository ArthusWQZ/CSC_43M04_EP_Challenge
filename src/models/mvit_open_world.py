from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


class MViTOpenWorld(nn.Module):
    """
    MViT-v2-S pretrained on Kinetics-400, fine-tuned for 33-class action recognition.

    Two-phase training (controlled by train.py via freeze_epochs config):
      Phase 1 (frozen backbone): only the new classification head is trained — fast,
        aligns the randomly-initialised head before touching pretrained weights.
      Phase 2 (last block unfrozen): the final transformer block + norm + head are
        fine-tuned at a low LR, letting temporal features adapt to the new classes.

    Input: (B, T, C, H, W) — standard dataset format, permuted internally.
    Requires T=16 and H=W=224 to match Kinetics pretraining conditions.
    Requires Kinetics normalisation (mean=0.45, std=0.225).
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = mvit_v2_s(weights=weights)
        model.head[1] = nn.Linear(model.head[1].in_features, num_classes)
        self.model = model

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        # Dataset gives (B, T, C, H, W); MViT expects (B, C, T, H, W)
        return self.model(video_batch.permute(0, 2, 1, 3, 4).contiguous())

    def freeze_backbone(self) -> None:
        """Freeze everything except the classifier head."""
        for name, param in self.model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    def unfreeze_last_block(self) -> None:
        """Unfreeze the final transformer block and the layer norm before the head."""
        last = len(self.model.blocks) - 1
        for name, param in self.model.named_parameters():
            if f"blocks.{last}." in name or name.startswith("norm") or name.startswith("head"):
                param.requires_grad = True
