"""
utils/label_utils.py
--------------------
Utility for managing class labels for the HAM10000 skin cancer dataset.

TWO uses:
    1. Run as a script (python utils/label_utils.py) to generate
       artifacts/class_labels.json — do this ONCE before starting the API
       when using the "weights" load strategy.
    2. Imported by model_loader.py / services to resolve short-codes.

HAM10000 canonical labels (7 classes):
    akiec  → Actinic keratoses
    bcc    → Basal cell carcinoma
    bkl    → Benign keratosis
    df     → Dermatofibroma
    mel    → Melanoma
    nv     → Melanocytic nevi
    vasc   → Vascular lesions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Ground truth mapping (from the notebook's lesion_type_dict) ──────────────
HAM10000_CODE_MAP: Dict[str, str] = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}

# Alphabetically sorted label list (matches FastAI's CategoryList ordering)
# FastAI sorts classes alphabetically when building the vocab from a DataFrame.
HAM10000_LABELS: List[str] = sorted(HAM10000_CODE_MAP.values())

# Reverse map: full label → short code
_LABEL_TO_CODE: Dict[str, str] = {v: k for k, v in HAM10000_CODE_MAP.items()}


def get_short_code(label: str) -> str:
    """Return the dataset short-code for a full label name."""
    return _LABEL_TO_CODE.get(label, "unknown")


def get_full_label(code: str) -> str:
    """Return the full label name for a dataset short-code."""
    return HAM10000_CODE_MAP.get(code.lower(), code)


def build_class_labels_json(output_path: Path) -> None:
    """
    Write artifacts/class_labels.json.
    Call this once from the command line after training.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": HAM10000_LABELS,
        "code_map": HAM10000_CODE_MAP,
        "num_classes": len(HAM10000_LABELS),
        "dataset": "HAM10000",
        "note": (
            "labels list is alphabetically sorted to match FastAI's "
            "CategoryList / vocab ordering."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✓  Wrote {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/class_labels.json")
    build_class_labels_json(out)
    print(f"Labels: {HAM10000_LABELS}")
