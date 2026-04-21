"""
services/model_service.py
-------------------------
ModelService is a singleton-style class that:
    1. Loads all model artifacts on startup (via load_model()).
    2. Exposes a predict() method that runs the full inference pipeline:
           raw bytes → preprocess → forward pass → softmax → top-k results
    3. Is stored on app.state so every request handler can access it
       without re-loading the model.

Design notes:
    - No training logic lives here — only inference.
    - Device management (CPU / CUDA) is handled transparently.
    - Thread-safety: PyTorch inference with torch.no_grad() is safe for
      concurrent reads; no mutable state is modified after load_model().
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from config.settings import settings
from models.model_loader import load_model_artifacts
from utils.image_preprocessor import preprocess_image_bytes
from utils.label_utils import get_short_code

logger = logging.getLogger(__name__)


class ModelService:
    """Encapsulates model lifecycle and inference."""

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._learner: Optional[Any] = None          # fastai Learner or None
        self._labels: List[str] = []                 # ordered class names
        self._code_map: Dict[str, str] = {}          # short_code → full_label
        self._device: Optional[torch.device] = None
        self._strategy: str = settings.MODEL_LOAD_STRATEGY
        self._loaded: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load model artifacts from disk.
        Called once at application startup (via lifespan in main.py).
        """
        logger.info(f"Loading model — strategy: '{self._strategy}'")
        t0 = time.perf_counter()

        artifacts = load_model_artifacts()

        self._model   = artifacts["model"]
        self._learner = artifacts["learner"]
        self._labels  = artifacts["labels"]
        self._code_map = artifacts["code_map"]
        self._device  = artifacts["device"]
        self._strategy = artifacts["strategy"]
        self._loaded  = True

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Model ready in {elapsed_ms} ms | "
            f"classes={len(self._labels)} | device={self._device}"
        )

    def unload_model(self) -> None:
        """Release GPU/CPU memory on shutdown."""
        self._model = None
        self._learner = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image_bytes: bytes,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on raw image bytes.

        Args:
            image_bytes : Raw bytes from an uploaded file.
            top_k       : Number of top predictions to return.
                          Defaults to settings.TOP_K_PREDICTIONS.

        Returns:
            Dict with keys:
                predicted_class     : str
                predicted_short_code: str
                confidence          : float
                confidence_pct      : str
                top_predictions     : List[Dict]
                inference_time_ms   : float
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Check startup logs for errors."
            )

        k = top_k or settings.TOP_K_PREDICTIONS
        t0 = time.perf_counter()

        # ── 1. Preprocess ──────────────────────────────────────────────────────
        tensor = preprocess_image_bytes(image_bytes)   # [1, C, H, W]
        tensor = tensor.to(self._device)

        # ── 2. Forward pass ────────────────────────────────────────────────────
        with torch.no_grad():
            logits = self._model(tensor)               # [1, num_classes]
            probabilities = F.softmax(logits, dim=1)   # [1, num_classes]

        probs_np = probabilities.squeeze(0).cpu().numpy()   # [num_classes]

        # ── 3. Build top-k results ─────────────────────────────────────────────
        top_indices = probs_np.argsort()[::-1][:k]
        top_predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            label = self._labels[idx]
            conf  = float(probs_np[idx])
            top_predictions.append(
                {
                    "rank":           rank,
                    "label":          label,
                    "short_code":     self._get_code_for_label(label),
                    "confidence":     round(conf, 6),
                    "confidence_pct": f"{conf * 100:.2f}%",
                }
            )

        best = top_predictions[0]
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            f"Prediction: '{best['label']}' ({best['confidence_pct']}) "
            f"in {inference_time_ms} ms"
        )

        return {
            "predicted_class":      best["label"],
            "predicted_short_code": best["short_code"],
            "confidence":           best["confidence"],
            "confidence_pct":       best["confidence_pct"],
            "top_predictions":      top_predictions,
            "inference_time_ms":    inference_time_ms,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_code_for_label(self, label: str) -> str:
        """Reverse-lookup the short code for a full label string."""
        # Try code_map reverse
        for code, full in self._code_map.items():
            if full == label:
                return code
        # Fallback: use label_utils
        return get_short_code(label)

    def get_class_info(self) -> List[Dict[str, str]]:
        """Return all classes with their short codes."""
        return [
            {
                "label":      label,
                "short_code": self._get_code_for_label(label),
            }
            for label in self._labels
        ]

    @property
    def strategy(self) -> str:
        return self._strategy
