"""
utils/file_validator.py
-----------------------
Validates uploaded image files before they reach the model.
Checks:
    - File size limit
    - MIME / content-type
    - Magic bytes (actual file content, not just extension)
"""

from __future__ import annotations

import logging
from fastapi import UploadFile, HTTPException, status
from config.settings import settings

logger = logging.getLogger(__name__)

# Magic bytes for allowed image formats
_MAGIC_BYTES: dict[str, bytes] = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png":  b"\x89PNG",
    "image/webp": b"RIFF",    # checked more precisely below
    "image/bmp":  b"BM",
}


async def validate_image_upload(file: UploadFile) -> bytes:
    """
    Read and validate an uploaded image file.

    Args:
        file: FastAPI UploadFile object.

    Returns:
        Raw bytes of the file (already read — caller must not re-read).

    Raises:
        HTTPException 400/413 on validation failure.
    """
    # ── 1. Content-type header check ─────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Allowed: {settings.ALLOWED_IMAGE_TYPES}"
            ),
        )

    # ── 2. Read bytes ─────────────────────────────────────────────────────────
    raw_bytes = await file.read()

    # ── 3. Size check ─────────────────────────────────────────────────────────
    size = len(raw_bytes)
    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if size > settings.MAX_UPLOAD_SIZE_BYTES:
        limit_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size / (1024*1024):.2f} MB exceeds limit of {limit_mb:.0f} MB.",
        )

    # ── 4. Magic bytes check ──────────────────────────────────────────────────
    _validate_magic_bytes(raw_bytes, content_type)

    logger.debug(
        f"File '{file.filename}' validated: {size} bytes, type={content_type}"
    )
    return raw_bytes


def _validate_magic_bytes(data: bytes, content_type: str) -> None:
    """Raise HTTPException if file magic bytes don't match the declared type."""
    expected = _MAGIC_BYTES.get(content_type)
    if expected is None:
        return  # Unknown type — already filtered above

    if not data[: len(expected)].startswith(expected):
        # WebP special: bytes 8-12 must be 'WEBP'
        if content_type == "image/webp" and data[8:12] == b"WEBP":
            return
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "File content does not match the declared content type. "
                "Please upload a valid image."
            ),
        )
