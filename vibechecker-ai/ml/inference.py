"""
VibeChecker AI — Inference Module
This is the public interface the backend (Zem) calls to get emotion predictions.

The module handles:
  - Loading the trained model once at startup (singleton pattern)
  - Preprocessing raw images / numpy arrays / file paths
  - Returning a structured prediction dict that matches the DB schema

Usage (from backend):
    from ml.inference import get_predictor

    predictor = get_predictor()                         # Loads model once
    result = predictor.predict_from_path("selfie.jpg")  # Predict from file
    result = predictor.predict_from_array(np_array)     # Predict from numpy array
    # result = {
    #   "emotion": "happy",
    #   "confidence": 0.82,
    #   "scores": {"angry": 0.02, "disgust": 0.01, ..., "happy": 0.82, ...},
    #   "model_version": "v1.0"
    # }
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import EMOTIONS, IMG_SIZE, EmotionCNN, load_model

# ── Normalisation must match what was used during training ───────────────────
_EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# Default model path — override via env var ML_MODEL_PATH or constructor arg
_DEFAULT_MODEL_PATH = str(
    Path(__file__).parent / "models" / "emotion_model_v1.0.pt"
)


class EmotionPredictor:
    """
    Wraps the trained EmotionCNN for single-image inference.
    Instantiate once and reuse — loading the model is expensive.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_version: str = "v1.0",
    ):
        """
        model_path    - Path to .pt checkpoint. Defaults to ML_MODEL_PATH env var
                        or models/emotion_model_v1.0.pt
        device        - 'cpu', 'cuda', or 'mps'. Auto-detected if None.
        model_version - Version string stored with predictions in the database.
        """
        self.model_version = model_version

        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Resolve model path
        if model_path is None:
            model_path = os.environ.get("ML_MODEL_PATH", _DEFAULT_MODEL_PATH)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                "Train the model first with:  python ml/train.py\n"
                "Or set ML_MODEL_PATH to point to an existing checkpoint."
            )

        self.model = load_model(model_path, device=self.device)
        print(f"[EmotionPredictor] Loaded model from {model_path} on {self.device}")

    # ── Public API ────────────────────────────────────────────

    def predict_from_path(self, image_path: str) -> dict:
        """
        Predict emotion from an image file path.

        image_path - Path to a JPEG, PNG, or any PIL-readable image
        Returns a prediction dict (see module docstring for schema).
        """
        img = Image.open(image_path).convert("RGB")
        return self._predict_pil(img)

    def predict_from_array(self, array: np.ndarray) -> dict:
        """
        Predict emotion from a numpy array (e.g., from OpenCV or MediaPipe).

        array - (H, W) grayscale or (H, W, 3) BGR/RGB uint8 array
        Returns a prediction dict.
        """
        # OpenCV uses BGR; PIL expects RGB — convert if 3-channel
        if array.ndim == 3 and array.shape[2] == 3:
            img = Image.fromarray(array[:, :, ::-1])  # BGR → RGB
        elif array.ndim == 2:
            img = Image.fromarray(array).convert("RGB")
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        return self._predict_pil(img)

    def predict_from_pil(self, img: Image.Image) -> dict:
        """
        Predict emotion from a PIL Image object.
        Returns a prediction dict.
        """
        return self._predict_pil(img)

    # ── Internal helpers ─────────────────────────────────────

    def _predict_pil(self, img: Image.Image) -> dict:
        """
        Core prediction pipeline:
          1. Apply eval transforms (grayscale, resize, normalise)
          2. Add batch dimension and move to device
          3. Forward pass → softmax probabilities
          4. Build result dict
        """
        tensor = _EVAL_TRANSFORM(img)            # (1, H, W)
        tensor = tensor.unsqueeze(0)             # (1, 1, H, W) — batch size 1
        tensor = tensor.to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(tensor)  # (1, NUM_CLASSES)

        probs_np = probs.squeeze(0).cpu().numpy()     # (NUM_CLASSES,)
        pred_idx = int(probs_np.argmax())
        pred_emotion = EMOTIONS[pred_idx]
        confidence = float(probs_np[pred_idx])

        scores = {emotion: round(float(p), 4) for emotion, p in zip(EMOTIONS, probs_np)}

        return {
            "emotion":       pred_emotion,
            "confidence":    round(confidence, 4),
            "scores":        scores,
            "model_version": self.model_version,
        }


# ── Singleton accessor ────────────────────────────────────────────────────────
# The backend calls get_predictor() to get a shared instance.
# The model is loaded only once, no matter how many times get_predictor() is called.

_predictor_instance: Optional[EmotionPredictor] = None


def get_predictor(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> EmotionPredictor:
    """
    Return the shared EmotionPredictor instance (lazy-initialised).
    Thread-safe for read-only inference.

    Example:
        from ml.inference import get_predictor
        predictor = get_predictor()
        result = predictor.predict_from_path("uploads/user1/selfie.jpg")
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EmotionPredictor(model_path=model_path, device=device)
    return _predictor_instance


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predictor = get_predictor()
    result = predictor.predict_from_path(image_path)

    print(f"\nPrediction for: {image_path}")
    print(f"  Emotion:    {result['emotion']}")
    print(f"  Confidence: {result['confidence']:.4f}  ({result['confidence']*100:.1f}%)")
    print(f"  All scores:")
    for emotion, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"    {emotion:<10} {score:.4f}  {bar}")
