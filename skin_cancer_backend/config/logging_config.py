"""
config/logging_config.py
------------------------
Structured logging setup.
Logs to both console (with colour) and a rotating file under LOG_DIR.
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging() -> None:
    """Configure root logger with console + rotating-file handlers."""
    from config.settings import settings

    log_dir: Path = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(settings.LOG_LEVEL)

    # ── Rotating file handler (10 MB × 5 backups) ─────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.LOG_LEVEL)

    # Remove any handlers already attached (avoids duplicate logs on reload)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
