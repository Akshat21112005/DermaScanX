"""
models/model_loader.py
----------------------
Handles loading of trained model artifacts from disk.
Supports TWO strategies depending on what Google Colab exported:

  Strategy A — "pkl"
      learn_eff.export('efficientnet-b3-deployment.pkl')
      A FastAI Learner export that bundles architecture + weights + class list.
      → Load with fastai.learner.load_learner()

  Strategy B — "weights"
      learn_eff.save('efficientnet-b3-weights')  →  produces .pth file
      Raw model state_dict saved by FastAI/PyTorch.
      → Recreate architecture with timm, then load_state_dict()

The strategy is controlled by MODEL_LOAD_STRATEGY in .env / settings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from config.settings import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")
    return device


def _load_class_labels_json(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Load class_labels.json produced by utils/label_utils.py.
    Returns (ordered_labels_list, short_code_to_label_dict).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels: List[str] = data["labels"]            # e.g. ["Actinic keratoses", ...]
    code_map: Dict[str, str] = data["code_map"]   # e.g. {"akiec": "Actinic keratoses", ...}
    logger.info(f"Loaded {len(labels)} class labels from {path}")
    return labels, code_map


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — FastAI .pkl
# ─────────────────────────────────────────────────────────────────────────────

def load_fastai_pkl(
    pkl_path: Path,
) -> Tuple[Any, List[str], Dict[str, str]]:
    """
    Load a FastAI Learner exported with learn.export().

    Returns:
        learner  – fastai Learner object (model inside .model)
        labels   – ordered list of class names  (from learner.dls.vocab)
        code_map – short-code → full-name mapping (loaded from JSON if available)
    """
    try:
        from fastai.learner import load_learner  # type: ignore
    except ImportError as e:
        raise ImportError(
            "fastai is not installed. Install it with:  pip install fastai"
        ) from e

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"FastAI .pkl not found at: {pkl_path}\n"
            "Copy the file exported by Colab (learn.export(...)) to this path."
        )

    logger.info(f"Loading FastAI learner from: {pkl_path}")
    learner = load_learner(pkl_path, cpu=not torch.cuda.is_available())
    learner.model.eval()

    # vocab is the class list in the order the model outputs probabilities
    labels: List[str] = list(learner.dls.vocab)
    logger.info(f"Model classes from FastAI vocab: {labels}")

    # Load code_map from JSON (optional but recommended)
    code_map: Dict[str, str] = {}
    if settings.CLASS_LABELS_PATH.exists():
        _, code_map = _load_class_labels_json(settings.CLASS_LABELS_PATH)

    return learner, labels, code_map


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B — Raw PyTorch weights (.pth)
# ─────────────────────────────────────────────────────────────────────────────

def load_pytorch_weights(
    weights_path: Path,
    labels_path: Path,
) -> Tuple[torch.nn.Module, List[str], Dict[str, str]]:
    """
    Recreate the EfficientNet-B3 architecture with timm and load saved weights.

    Returns:
        model    – torch.nn.Module in eval mode
        labels   – ordered list of class names
        code_map – short-code → full-name mapping
    """
    try:
        import timm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "timm is not installed. Install it with:  pip install timm"
        ) from e

    # ── 1. Load class labels (required for weights strategy) ─────────────────
    if not labels_path.exists():
        raise FileNotFoundError(
            f"class_labels.json not found at: {labels_path}\n"
            "Generate it with:  python utils/label_utils.py\n"
            "or copy it from your Colab environment."
        )
    labels, code_map = _load_class_labels_json(labels_path)
    num_classes = len(labels)

    # ── 2. Recreate architecture ──────────────────────────────────────────────
    logger.info(
        f"Creating timm model: {settings.TIMM_MODEL_NAME} "
        f"with {num_classes} output classes"
    )
    model = timm.create_model(
        settings.TIMM_MODEL_NAME,
        pretrained=False,       # weights will be loaded from .pth
        num_classes=num_classes,
    )

    # ── 3. Load weights ───────────────────────────────────────────────────────
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found at: {weights_path}\n"
            "Copy the .pth file generated by Colab (learn.save(...)) to this path."
        )

    device = _resolve_device()
    logger.info(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # FastAI save() wraps the state_dict under a 'model' key in some versions
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    logger.info("Weights loaded successfully.")

    return model, labels, code_map


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_model_artifacts() -> Dict[str, Any]:
    """
    Load model and class metadata according to MODEL_LOAD_STRATEGY.

    Returns a dict with keys:
        strategy  : str
        model     : nn.Module  (always a bare PyTorch module)
        learner   : fastai Learner | None
        labels    : List[str]
        code_map  : Dict[str, str]
        device    : torch.device
    """
    strategy = settings.MODEL_LOAD_STRATEGY
    device = _resolve_device()

    if strategy == "pkl":
        learner, labels, code_map = load_fastai_pkl(settings.MODEL_PKL_PATH)
        return {
            "strategy": "pkl",
            "learner": learner,
            "model": learner.model,
            "labels": labels,
            "code_map": code_map,
            "device": device,
        }

    elif strategy == "weights":
        model, labels, code_map = load_pytorch_weights(
            weights_path=settings.MODEL_WEIGHTS_PATH,
            labels_path=settings.CLASS_LABELS_PATH,
        )
        return {
            "strategy": "weights",
            "learner": None,
            "model": model,
            "labels": labels,
            "code_map": code_map,
            "device": device,
        }

    else:
        raise ValueError(f"Unknown MODEL_LOAD_STRATEGY: '{strategy}'")
