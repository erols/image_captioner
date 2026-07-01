"""Resize images for VLM input."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def resize_for_vlm(
    src_path: Path, dest_path: Path, max_dim: int = 1568, quality: int = 92
) -> None:
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, max_dim / max(width, height))
        if scale < 1.0:
            img = img.resize(
                (round(width * scale), round(height * scale)), Image.LANCZOS
            )
        img.save(dest_path, format="JPEG", quality=quality)
