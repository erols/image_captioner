"""Shared test helpers for synthesizing sample images."""
from pathlib import Path

from PIL import Image


def make_solid_image(
    path: Path, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path)
