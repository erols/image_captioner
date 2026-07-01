from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import rawpy
from PIL import Image, ImageDraw

from image_captioner.hashing import content_hash, hamming_distance, perceptual_hash
from tests.helpers import make_solid_image


def test_content_hash_deterministic_and_sensitive_to_bytes(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    make_solid_image(path_a, (100, 100), (255, 0, 0))
    make_solid_image(path_b, (100, 100), (0, 255, 0))

    assert content_hash(path_a) == content_hash(path_a)
    assert content_hash(path_a) != content_hash(path_b)


def test_perceptual_hash_similar_for_resized_recompressed_copy(tmp_path: Path) -> None:
    original = tmp_path / "original.jpg"
    make_solid_image(original, (800, 600), (30, 60, 90))

    resized = tmp_path / "resized.jpg"
    with Image.open(original) as img:
        img.resize((200, 150)).save(resized, quality=80)

    distant = tmp_path / "distant.jpg"
    img = Image.new("RGB", (800, 600), (220, 20, 140))
    draw = ImageDraw.Draw(img)
    # Add deterministic structure (stripe pattern) to make it perceptually distinct
    for i in range(0, 800, 80):
        draw.rectangle([(i, 0), (i + 40, 600)], fill=(255, 100, 180))
    img.save(distant)

    near_distance = hamming_distance(perceptual_hash(original), perceptual_hash(resized))
    far_distance = hamming_distance(perceptual_hash(original), perceptual_hash(distant))

    assert near_distance <= 5
    assert far_distance > near_distance


def test_perceptual_hash_falls_back_when_raw_has_no_thumbnail(tmp_path: Path) -> None:
    # Some RAW files (e.g. certain phone-camera DNGs) have no thumbnail libraw
    # can extract at all; perceptual_hash must still succeed for these.
    raw_path = tmp_path / "no_thumb.dng"
    raw_path.write_bytes(b"fake-raw-bytes")

    fake_raw = MagicMock()
    fake_raw.extract_thumb.side_effect = rawpy.LibRawNoThumbnailError("no thumbnail")
    fake_raw.postprocess.return_value = np.zeros((10, 10, 3), dtype="uint8")
    fake_raw.__enter__ = MagicMock(return_value=fake_raw)
    fake_raw.__exit__ = MagicMock(return_value=False)

    with patch("image_captioner.hashing.rawpy.imread", return_value=fake_raw):
        result = perceptual_hash(raw_path)

    assert isinstance(result, str)
    fake_raw.postprocess.assert_called_once_with(half_size=True, use_camera_wb=True)
