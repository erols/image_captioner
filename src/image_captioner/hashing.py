"""Content and perceptual hashing utilities."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import imagehash
import rawpy
from PIL import Image

from image_captioner.formats import RAW_EXTENSIONS


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_for_hash(path: Path) -> Image.Image:
    if path.suffix.lower() in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                # Some RAW files (notably certain phone-camera DNGs) don't embed
                # a thumbnail libraw can extract. Fall back to a fast half-size
                # decode of the full raw data purely for hashing purposes.
                rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                return Image.fromarray(rgb)
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(thumb.data))
        return Image.fromarray(thumb.data)
    return Image.open(path)


def perceptual_hash(path: Path) -> str:
    with _load_for_hash(path) as img:
        return str(imagehash.phash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
