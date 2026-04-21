"""
VibeChecker AI — FER2013 Dataset Loader
Loads pre-split FER2013 images from the directory structure Javaya prepares:

    data/
      train/
        angry/   disgust/  fear/  happy/  sad/  surprise/  neutral/
      val/
        angry/   ...
      test/
        angry/   ...

Each subfolder contains raw .jpg/.png face images (typically 48x48 grayscale,
but the loader handles any size and converts to grayscale automatically).

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders("data/")
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from model import EMOTIONS, IMG_SIZE

# ── ImageNet-style normalisation for grayscale ───────────────────────────────
# FER2013 pixel values range 0-255; we normalize to mean≈0.5, std≈0.5
MEAN = (0.5,)
STD  = (0.5,)


def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Aggressive enough to reduce overfitting on the small FER2013 dataset,
    but conservative enough to preserve facial expressions.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),         # Ensure single channel
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),              # Faces are symmetric
        transforms.RandomRotation(degrees=10),               # Small head tilt
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Minor shift
        transforms.ColorJitter(brightness=0.3, contrast=0.3),     # Lighting variation
        transforms.ToTensor(),                               # → [0,1] float tensor
        transforms.Normalize(mean=MEAN, std=STD),            # → ~[-1,1]
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Occlusion simulation
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Minimal transforms for val/test — no augmentation, just resize + normalize.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to address FER2013's class imbalance.
    FER2013 has ~8x more 'happy' samples than 'disgust'; this sampler ensures
    each class is seen roughly equally during training.
    """
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    # Weight for each sample = 1 / count of its class
    weights = [1.0 / class_counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from a FER2013-structured directory.

    data_dir             - Root folder containing train/, val/, test/ subfolders
    batch_size           - Samples per batch (64 works well for FER2013 on GPU)
    num_workers          - Parallel workers for data loading (0 on Windows if issues)
    use_weighted_sampler - Balance classes during training (recommended)

    Returns (train_loader, val_loader, test_loader).
    """
    data_path = Path(data_dir)

    # ── Datasets ─────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(
        root=str(data_path / "train"),
        transform=get_train_transforms(),
    )
    val_dataset = datasets.ImageFolder(
        root=str(data_path / "val"),
        transform=get_eval_transforms(),
    )
    test_dataset = datasets.ImageFolder(
        root=str(data_path / "test"),
        transform=get_eval_transforms(),
    )

    # Verify class order matches EMOTIONS list
    assert train_dataset.classes == EMOTIONS, (
        f"Dataset class order {train_dataset.classes} does not match "
        f"expected EMOTIONS {EMOTIONS}. Check your data directory structure."
    )

    # ── Samplers ─────────────────────────────────────────────
    train_sampler = make_weighted_sampler(train_dataset) if use_weighted_sampler else None

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),   # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,    # Avoid tiny last batch messing up BatchNorm
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Print a summary so the user can verify everything loaded correctly
    print(f"Dataset loaded from: {data_path.resolve()}")
    print(f"  Train: {len(train_dataset):>6,} images")
    print(f"  Val:   {len(val_dataset):>6,} images")
    print(f"  Test:  {len(test_dataset):>6,} images")
    print(f"  Classes ({len(train_dataset.classes)}): {train_dataset.classes}")

    return train_loader, val_loader, test_loader


def get_class_weights(data_dir: str, device: str = "cpu") -> torch.Tensor:
    """
    Compute per-class weights for use in a weighted CrossEntropyLoss.
    This is an alternative to WeightedRandomSampler — use one or the other.

    Returns a (NUM_CLASSES,) tensor with inverse-frequency weights.
    """
    train_dataset = datasets.ImageFolder(
        root=str(Path(data_dir) / "train"),
        transform=get_eval_transforms(),
    )

    class_counts = torch.zeros(len(train_dataset.classes))
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    # Inverse frequency, normalised so the median class has weight 1.0
    weights = 1.0 / class_counts
    weights = weights / weights.median()
    return weights.to(device)
