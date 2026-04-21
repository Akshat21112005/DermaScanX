"""
utils/image_preprocessor.py
----------------------------
Reusable image preprocessing pipeline that EXACTLY mirrors
the transformations applied during training in the Colab notebook:

    - Resize to IMAGE_SIZE × IMAGE_SIZE  (224 × 224 by default)
    - Convert to RGB  (handles grayscale / RGBA uploads)
    - ToTensor         [H, W, C] uint8 → [C, H, W] float32 in [0, 1]
    - Normalize with ImageNet mean/std  (same stats used in .normalize(imagenet_stats))

All logic lives here so any future change to preprocessing is made
in ONE place and is immediately reflected in every endpoint.
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Build the transform pipeline once at import time
# ─────────────────────────────────────────────────────────────────────────────

def _build_transform() -> T.Compose:
    """
    Construct the inference-time transform pipeline.
    Called once; result cached as module-level constant.
    """
    return T.Compose(
        [
            T.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            T.ToTensor(),           # → [C, H, W] float32 in [0, 1]
            T.Normalize(
                mean=settings.IMAGENET_MEAN,
                std=settings.IMAGENET_STD,
            ),
        ]
    )


# Module-level singleton — avoids rebuilding on every request
INFERENCE_TRANSFORM: T.Compose = _build_transform()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Accept raw image bytes (from an uploaded file) and return a
    normalised tensor ready for model inference.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        torch.Tensor of shape [1, C, H, W] (batch dimension added).

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"Cannot decode image: {e}") from e

    # Ensure 3-channel RGB regardless of upload format
    if img.mode != "RGB":
        logger.debug(f"Converting image from mode '{img.mode}' to 'RGB'")
        img = img.convert("RGB")

    tensor: torch.Tensor = INFERENCE_TRANSFORM(img)   # [C, H, W]
    return tensor.unsqueeze(0)                          # [1, C, H, W]


def preprocess_pil_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Accept a PIL Image and return a batch tensor.
    Useful in testing / notebook usage.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return INFERENCE_TRANSFORM(pil_image).unsqueeze(0)


def get_transform_description() -> dict:
    """Return a human-readable description of the preprocessing steps."""
    return {
        "resize": f"{settings.IMAGE_SIZE}x{settings.IMAGE_SIZE}",
        "normalisation": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        "colour_mode": "RGB",
        "output_shape": f"[1, 3, {settings.IMAGE_SIZE}, {settings.IMAGE_SIZE}]",
    }
