"""
VibeChecker AI — Model Training Script
Trains the EmotionCNN on FER2013 data with:
  - Cosine annealing LR schedule with warm restarts
  - Label smoothing cross-entropy loss
  - Early stopping on validation accuracy
  - Checkpoint saving (best model + latest epoch)
  - Training curve logging to CSV

Usage:
    python train.py
    python train.py --data data/ --epochs 60 --batch-size 64 --lr 1e-3
    python train.py --resume models/emotion_model_v1.0.pt
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import get_dataloaders
from model import EmotionCNN, NUM_CLASSES


# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULTS = {
    "data_dir":    "data/",
    "model_dir":   "models/",
    "epochs":      60,
    "batch_size":  64,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "patience":    10,          # Early stopping: stop after N epochs with no improvement
    "model_version": "v1.0",
    "num_workers": 4,
}


# ═════════════════════════════════════════════════════════════════════════════
# Training & Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: EmotionCNN,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler=None,
) -> tuple[float, float]:
    """
    Run one full training epoch.
    Returns (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step scheduler every batch (CosineAnnealingWarmRestarts expects this)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: EmotionCNN,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    Evaluate model on a validation or test loader.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model: EmotionCNN, optimizer, epoch: int, val_acc: float,
                    path: str, version: str):
    """Save a full checkpoint so training can be resumed later."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "version": version,
    }, path)


def get_device() -> str:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═════════════════════════════════════════════════════════════════════════════

def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # ── Directories ──────────────────────────────────────────
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path   = model_dir / f"emotion_model_{args.model_version}.pt"
    latest_model_path = model_dir / "emotion_model_latest.pt"
    log_path          = model_dir / "training_log.csv"

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ────────────────────────────────────────────────
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"  Resuming at epoch {start_epoch}, best val acc so far: {best_val_acc:.4f}")

    # ── Loss: label smoothing helps with FER2013's noisy labels ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer: AdamW with weight decay ───────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ── LR scheduler: cosine warm restarts ───────────────────
    # T_0 = restart period in steps; set to 1 epoch worth of steps
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch * 10, T_mult=1, eta_min=1e-6
    )

    # ── CSV log ───────────────────────────────────────────────
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_s"])

    # ── Early stopping state ─────────────────────────────────
    epochs_no_improve = 0

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  Best model → {best_model_path}")
    print(f"  Log        → {log_path}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  |  "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.2e}  |  {elapsed:.1f}s"
        )

        log_writer.writerow([epoch+1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                              f"{val_loss:.6f}", f"{val_acc:.6f}",
                              f"{current_lr:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        # Always save the latest checkpoint so training can be resumed
        save_checkpoint(model, optimizer, epoch, val_acc,
                        str(latest_model_path), args.model_version)

        # Save best model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_acc,
                            str(best_model_path), args.model_version)
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                break

    log_file.close()

    # ── Final test evaluation ─────────────────────────────────
    print(f"\nLoading best model for final test evaluation: {best_model_path}")
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train VibeChecker emotion model on FER2013")
    p.add_argument("--data",          type=str,   default=DEFAULTS["data_dir"],      help="Path to data/ directory")
    p.add_argument("--model-dir",     type=str,   default=DEFAULTS["model_dir"],     help="Where to save checkpoints")
    p.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"],        help="Max training epochs")
    p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"],    help="Batch size")
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"],            help="Initial learning rate")
    p.add_argument("--weight-decay",  type=float, default=DEFAULTS["weight_decay"],  help="AdamW weight decay")
    p.add_argument("--patience",      type=int,   default=DEFAULTS["patience"],      help="Early stopping patience")
    p.add_argument("--model-version", type=str,   default=DEFAULTS["model_version"], help="Version tag for saved model")
    p.add_argument("--num-workers",   type=int,   default=DEFAULTS["num_workers"],   help="DataLoader workers")
    p.add_argument("--resume",        type=str,   default=None,                      help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Map argparse dashes to underscores for attribute access
    args.data_dir = args.data
    train(args)
