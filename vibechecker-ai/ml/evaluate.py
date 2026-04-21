"""
VibeChecker AI — Model Evaluation Script
Runs the trained model against the test split and prints:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Confusion matrix (text + optional PNG)

Usage:
    python evaluate.py
    python evaluate.py --checkpoint models/emotion_model_v1.0.pt --data data/
    python evaluate.py --checkpoint models/emotion_model_v1.0.pt --save-plot
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import get_dataloaders
from model import EMOTIONS, EmotionCNN, load_model


# ═════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_confusion_matrix(
    model: EmotionCNN, loader, device: str
) -> torch.Tensor:
    """
    Run the model over the loader and accumulate a confusion matrix.
    Returns a (NUM_CLASSES, NUM_CLASSES) int tensor where
    cm[true_label][pred_label] = count.
    """
    num_classes = len(EMOTIONS)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            for true, pred in zip(labels, preds):
                cm[true][pred] += 1

    return cm


def per_class_metrics(cm: torch.Tensor) -> dict:
    """
    Derive precision, recall, and F1 per class from the confusion matrix.
    Returns a dict keyed by emotion name.
    """
    metrics = {}
    for i, emotion in enumerate(EMOTIONS):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp   # Others predicted as class i
        fn = cm[i, :].sum().item() - tp   # Class i predicted as others

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = cm[i, :].sum().item()

        metrics[emotion] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   int(support),
        }
    return metrics


def print_report(cm: torch.Tensor, metrics: dict):
    """Print a formatted classification report + confusion matrix to stdout."""
    total = cm.sum().item()
    correct = cm.diagonal().sum().item()
    overall_acc = correct / total if total > 0 else 0.0

    print("\n" + "═" * 62)
    print(f"  Overall Accuracy: {overall_acc:.4f}  ({correct}/{total})")
    print("═" * 62)

    # Per-class table
    print(f"\n{'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 56)
    for emotion, m in metrics.items():
        print(
            f"{emotion:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10}"
        )

    # Macro averages
    macro_p = sum(m["precision"] for m in metrics.values()) / len(metrics)
    macro_r = sum(m["recall"]    for m in metrics.values()) / len(metrics)
    macro_f = sum(m["f1"]        for m in metrics.values()) / len(metrics)
    print("-" * 56)
    print(f"{'macro avg':<12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f}")

    # Confusion matrix
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':>12}" + "".join(f"{e[:7]:>9}" for e in EMOTIONS)
    print(header)
    for i, emotion in enumerate(EMOTIONS):
        row_vals = "".join(f"{cm[i, j].item():>9}" for j in range(len(EMOTIONS)))
        print(f"{emotion:<12}{row_vals}")
    print()


def save_confusion_matrix_plot(cm: torch.Tensor, output_path: str):
    """
    Save a colour-coded confusion matrix PNG using matplotlib.
    Only runs if matplotlib is installed — skips gracefully if not.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        cm_np = cm.numpy().astype(float)
        # Normalise rows to show proportions
        row_sums = cm_np.sum(axis=1, keepdims=True)
        cm_norm = cm_np / (row_sums + 1e-8)

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_yticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
        ax.set_yticklabels(EMOTIONS)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("VibeChecker — Emotion Recognition Confusion Matrix")

        # Annotate cells with raw counts
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(int(cm_np[i, j])),
                        ha="center", va="center", color=color, fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Confusion matrix plot saved to: {output_path}")

    except ImportError:
        print("matplotlib not installed — skipping confusion matrix plot.")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.checkpoint}")

    model = load_model(args.checkpoint, device=device)

    _, _, test_loader = get_dataloaders(
        args.data, batch_size=64, num_workers=args.num_workers
    )

    print("Running evaluation on test set...")
    cm = compute_confusion_matrix(model, test_loader, device)
    metrics = per_class_metrics(cm)
    print_report(cm, metrics)

    if args.save_plot:
        plot_path = str(Path(args.checkpoint).parent / "confusion_matrix.png")
        save_confusion_matrix_plot(cm, plot_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VibeChecker emotion model")
    p.add_argument("--checkpoint",   type=str, default="models/emotion_model_v1.0.pt",
                   help="Path to model checkpoint")
    p.add_argument("--data",         type=str, default="data/",
                   help="Path to data/ directory")
    p.add_argument("--num-workers",  type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--save-plot",    action="store_true",
                   help="Save confusion matrix as PNG")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
