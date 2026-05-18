"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm
    python train.py experiment=cnn_temporal_transformer

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
import numpy as np

from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_temporal_transformer import CNNTemporalTransformer
from models.efficientnet_temporal import EfficientNetTemporal
from models.mvit_open_world import MViTOpenWorld
from models.resnet50_baseline import ResNet50Baseline
from models.tsm_resnet18 import TSMResNet18
from utils import build_transforms, set_seed, split_train_val


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "cnn_temporal_transformer":
        return CNNTemporalTransformer(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=pretrained,
            embed_dim=int(cfg.model.get("embed_dim", 256)),
            num_heads=int(cfg.model.get("num_heads", 4)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            dropout=float(cfg.model.get("dropout", 0.2)),
        )
    if name == "resnet50_baseline":
        return ResNet50Baseline(num_classes=num_classes, pretrained=pretrained)
    if name == "tsm_resnet18":
        return TSMResNet18(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=pretrained,
            temporal_pool=str(cfg.model.get("temporal_pool", "last")),
        )
    if name == "mvit_open_world":
        return MViTOpenWorld(num_classes=num_classes, pretrained=pretrained)
    if name == "efficientnet_temporal":
        return EfficientNetTemporal(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=pretrained,
            embed_dim=int(cfg.model.get("embed_dim", 512)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            use_checkpoint=bool(cfg.model.get("use_checkpoint", True)),
        )

    raise ValueError(f"Unknown model.name: {name}")


def _mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns (mixed_x, y_a, y_b, lam) for a Mixup pass."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        if mixup_alpha > 0.0:
            video_batch, labels_a, labels_b, lam = _mixup_batch(video_batch, labels, mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(video_batch)
            loss = lam * loss_fn(logits, labels_a) + (1.0 - lam) * loss_fn(logits, labels_b)

        scaler.scale(loss).backward()

        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels_a).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # use TF32 on Ampere+ GPUs

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    use_imagenet_norm = bool(cfg.model.pretrained)
    augmentation = str(cfg.dataset.get("augmentation", "standard"))
    normalization = str(cfg.dataset.get("normalization", "imagenet" if use_imagenet_norm else "default"))
    train_transform = build_transforms(
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        augmentation=augmentation,
        normalization=normalization,
    )
    eval_transform = build_transforms(
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        augmentation=augmentation,
        normalization=normalization,
    )
    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    sampling = str(cfg.dataset.get("sampling", "uniform"))

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
        sampling=sampling,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
        sampling=sampling,
    )

    num_workers = int(cfg.training.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = build_model(cfg).to(device)

    if bool(cfg.training.get("use_compile", False)) and hasattr(torch, "compile"):
        print("Compiling model with torch.compile …")
        model = torch.compile(model)

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    base_lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    backbone_scale = float(cfg.training.get("lr_backbone_scale", 1.0))
    freeze_epochs = int(cfg.training.get("freeze_epochs", 0))

    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        print(f"Backbone frozen — head-only training for first {freeze_epochs} epochs.")

    if freeze_epochs == 0 and backbone_scale != 1.0 and hasattr(model, "param_groups"):
        optimizer = torch.optim.AdamW(
            model.param_groups(base_lr, backbone_scale), weight_decay=weight_decay
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)

    epochs = int(cfg.training.epochs)
    scheduler_name = cfg.training.get("scheduler", None)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    min_lr = float(cfg.training.get("min_lr", 1e-6))
    scheduler = None
    if scheduler_name == "cosine":
        cosine_epochs = max(1, epochs - warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine_sched

    use_amp = bool(cfg.training.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(cfg.training.get("grad_clip", 0.0))

    print(
        f"Training: epochs={epochs}, batch={cfg.training.batch_size}, "
        f"lr={cfg.training.lr}, amp={use_amp}, scheduler={scheduler_name}, "
        f"augmentation={augmentation}, mixup_alpha={mixup_alpha}"
    )

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            if hasattr(model, "unfreeze_last_block"):
                model.unfreeze_last_block()
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=base_lr * 0.1, weight_decay=weight_decay)
            remaining = epochs - epoch
            scheduler = (
                CosineAnnealingLR(optimizer, T_max=max(1, remaining), eta_min=min_lr)
                if scheduler_name == "cosine" else None
            )
            print(f"Epoch {epoch + 1}: last block unfrozen, LR={base_lr * 0.1:.1e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, grad_clip, mixup_alpha
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
