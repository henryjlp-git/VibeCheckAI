"""
VibeChecker AI — Emotion Classification Model
CNN architecture designed for FER2013 (48x48 grayscale face images).
7 output classes: angry, disgust, fear, happy, sad, surprise, neutral

Architecture overview:
  - 4 convolutional blocks with batch norm + dropout for regularization
  - Global average pooling instead of flatten to reduce overfitting
  - Two FC layers with dropout before final softmax output

Usage:
    from model import EmotionCNN, EMOTIONS
    model = EmotionCNN()
"""

import torch
import torch.nn as nn

# FER2013 emotion labels in the order used by the dataset
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)

# FER2013 image properties
IMG_SIZE = 48       # Width and height in pixels
IMG_CHANNELS = 1    # Grayscale


class ConvBlock(nn.Module):
    """
    Reusable conv block: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool → Dropout.
    Doubling the conv keeps spatial context before pooling halves the resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # Halves spatial dims
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    CNN for facial emotion recognition on 48x48 grayscale images.

    Feature map progression (H x W x C):
        Input:  48 x 48 x 1
        Block1: 24 x 24 x 64
        Block2: 12 x 12 x 128
        Block3:  6 x  6 x 256
        Block4:  3 x  3 x 512
        GAP:     1 x  1 x 512  → 512-dim vector
        FC1:     256
        FC2:     NUM_CLASSES (7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_fc: float = 0.5):
        super().__init__()

        # ── Convolutional feature extractor ─────────────────
        self.features = nn.Sequential(
            ConvBlock(IMG_CHANNELS, 64,  dropout=0.25),
            ConvBlock(64,           128, dropout=0.25),
            ConvBlock(128,          256, dropout=0.25),
            ConvBlock(256,          512, dropout=0.25),
        )

        # Global average pooling: collapses spatial dims to 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation — He init for ReLU activations
        self._init_weights()

    def _init_weights(self):
        """Apply He (Kaiming) initialisation to conv and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch, 1, 48, 48) float tensor, values in [0, 1]
        Returns: (batch, num_classes) raw logits — apply softmax for probabilities
        """
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method — returns softmax probabilities instead of logits.
        x: (batch, 1, 48, 48) float tensor
        Returns: (batch, num_classes) probability tensor
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def load_model(checkpoint_path: str, device: str = "cpu") -> EmotionCNN:
    """
    Load a saved model checkpoint.

    checkpoint_path - Path to a .pt file saved by train.py
    device          - 'cpu', 'cuda', or 'mps'

    Returns a model in eval mode, ready for inference.
    """
    model = EmotionCNN()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict saves and wrapped checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check — print the model and verify output shape
    model = EmotionCNN()
    dummy = torch.zeros(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)  # Batch of 4
    out = model(dummy)
    print(model)
    print(f"\nInput:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: (4, {NUM_CLASSES}))")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
