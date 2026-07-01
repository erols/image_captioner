"""Shared test helpers for synthesizing sample images."""
from pathlib import Path

from PIL import Image, ImageDraw


def make_solid_image(
    path: Path, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path)


def make_structured_image(
    path: Path, size: tuple[int, int], base_color: tuple[int, int, int], accent_color: tuple[int, int, int]
) -> None:
    """Create an image with visual structure (checkerboard pattern) to ensure a
    distinct perceptual hash from a solid-color image of similar brightness.

    A one-dimensional stripe pattern only varies the horizontal frequency
    content, which is not enough to reliably shift the phash's DCT-based
    bits far from a solid image's. A two-dimensional checkerboard varies
    both horizontal and vertical frequency content, producing a phash that
    differs from a solid image's by a wide, reliable margin.
    """
    width, height = size
    img = Image.new("RGB", size, base_color)
    draw = ImageDraw.Draw(img)
    tile = width // 8  # checkerboard tiles across both axes
    for ty in range(0, height, tile):
        for tx in range(0, width, tile):
            if ((tx // tile) + (ty // tile)) % 2 == 0:
                draw.rectangle([(tx, ty), (tx + tile, ty + tile)], fill=accent_color)
    img.save(path)
