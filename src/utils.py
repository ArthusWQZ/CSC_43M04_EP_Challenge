"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    augmentation: str = "standard",
    normalization: str = "imagenet",
) -> transforms.Compose:
    """
    Standard torchvision pipeline for single RGB frames.

    normalization:
        "imagenet" -> ImageNet mean/std (pretrained image models)
        "kinetics" -> Kinetics mean/std (required for MViT-v2 pretrained on Kinetics)
        "default"  -> [0.5, 0.5, 0.5] normalisation

    augmentation (training only):
        "standard"     -> Resize only (no horizontal flip: dataset has direction-sensitive classes)
        "strong"       -> RandomResizedCrop + ColorJitter + RandomGrayscale
        "very_strong"  -> strong + RandAugment + RandomErasing

    Note: RandomHorizontalFlip is intentionally absent from all modes.
    The dataset contains direction-sensitive action classes (e.g. "moving left" ≠
    "moving right"), and transforms are applied per-frame independently, so a
    consistent flip across frames cannot be guaranteed.  Enabling hflip would
    corrupt labels for direction-sensitive classes.
    """
    if normalization == "kinetics":
        normalize = transforms.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        )
    elif use_imagenet_norm or normalization == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        if augmentation == "very_strong":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandAugment(num_ops=2, magnitude=9),
                    transforms.ToTensor(),
                    normalize,
                    transforms.RandomErasing(p=0.25),
                ]
            )
        if augmentation == "strong":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        # "standard"
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # Eval: standard uses plain resize; strong/very_strong uses resize+centercrop (ImageNet protocol)
    if augmentation in ("strong", "very_strong"):
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    # (batch_size, max_k) indices of top predictions
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()  # (max_k, batch_size)
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        # Any hit in the top-k row slice counts
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Shuffle then split a list of (video_path, label) into train and validation portions.

    Mirrors a standard random hold-out so train.py and evaluate.py stay consistent.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples
