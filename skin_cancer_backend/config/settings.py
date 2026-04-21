"""
config/settings.py
------------------
Centralised configuration using pydantic-settings.
All values are read from environment variables (or .env file).
Override any value by setting the corresponding env var.
"""

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App meta ──────────────────────────────────────────────────────────────
    APP_NAME: str = "Skin Cancer Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("logs")

    # ── Model artifacts (paths provided manually after Colab training) ────────
    # These paths point to files you copy from Google Colab to your local machine.
    #
    # Option A – FastAI export (.pkl): single file that bundles model + weights + classes
    MODEL_PKL_PATH: Path = Path("artifacts/efficientnet-b3-deployment.pkl")
    #
    # Option B – Raw PyTorch weights (.pth / .pt): load with timm architecture
    MODEL_WEIGHTS_PATH: Path = Path("artifacts/efficientnet-b3-weights.pth")
    #
    # Class labels JSON  (generated automatically if not present)
    CLASS_LABELS_PATH: Path = Path("artifacts/class_labels.json")
    #
    # Which loading strategy to use: "pkl" | "weights"
    MODEL_LOAD_STRATEGY: str = "pkl"

    # ── Inference ─────────────────────────────────────────────────────────────
    # Image dimensions used during training
    IMAGE_SIZE: int = 224
    # ImageNet normalisation stats (same values used in training)
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]
    # Maximum upload size (bytes) — default 10 MB
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024
    # Allowed MIME types for uploaded images
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    # Number of top predictions to return
    TOP_K_PREDICTIONS: int = 3

    # ── timm model name (used only when MODEL_LOAD_STRATEGY == "weights") ─────
    TIMM_MODEL_NAME: str = "efficientnet_b3"
    NUM_CLASSES: int = 7

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @field_validator("MODEL_LOAD_STRATEGY")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"pkl", "weights"}
        if v.lower() not in allowed:
            raise ValueError(f"MODEL_LOAD_STRATEGY must be one of {allowed}")
        return v.lower()


# Singleton instance — import this everywhere
settings = Settings()
