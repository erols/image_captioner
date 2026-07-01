from pathlib import Path

from PIL import Image

from image_captioner.resize import resize_for_vlm
from tests.helpers import make_solid_image


def test_large_image_is_downscaled_to_max_dim(tmp_path: Path) -> None:
    src = tmp_path / "large.jpg"
    dest = tmp_path / "resized.jpg"
    make_solid_image(src, (3000, 2000), (100, 150, 200))

    resize_for_vlm(src, dest, max_dim=1568, quality=92)

    with Image.open(dest) as img:
        assert max(img.size) <= 1568
        # Check aspect ratio is preserved within floating-point tolerance
        # (integer rounding can't preserve exact ratios in all cases)
        assert abs(img.size[0] / img.size[1] - 3000 / 2000) < 0.001


def test_small_image_is_not_upscaled(tmp_path: Path) -> None:
    src = tmp_path / "small.jpg"
    dest = tmp_path / "resized.jpg"
    make_solid_image(src, (200, 100), (10, 10, 10))

    resize_for_vlm(src, dest, max_dim=1568, quality=92)

    with Image.open(dest) as img:
        assert img.size == (200, 100)
