from pathlib import Path
from unittest.mock import patch

from image_captioner.manifest import Manifest
from image_captioner.raw_convert import run_convert_raw
from tests.helpers import make_solid_image


def test_non_raw_images_pass_through_as_done(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    jpeg_path = input_dir / "photo.jpg"
    make_solid_image(jpeg_path, (200, 150), (10, 20, 30))
    raw_originals_dir = tmp_path / "_raw_originals"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(jpeg_path, "hash1")
        run_convert_raw(raw_originals_dir, manifest)

        record = manifest.get(str(jpeg_path))
        assert record.raw_status == "done"
        assert record.current_path == str(jpeg_path)
        assert jpeg_path.exists()
    finally:
        manifest.close()


def test_raw_image_converted_and_original_archived(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    raw_path = input_dir / "shot.cr2"
    raw_path.write_bytes(b"fake-raw-bytes")
    raw_originals_dir = tmp_path / "_raw_originals"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(raw_path, "hash1")

        def fake_convert(src: Path, dest: Path, quality: int = 95) -> None:
            make_solid_image(dest, (100, 100), (5, 5, 5))

        with patch("image_captioner.raw_convert.convert_raw_to_jpeg", side_effect=fake_convert):
            run_convert_raw(raw_originals_dir, manifest)

        record = manifest.get(str(raw_path))
        assert record.raw_status == "done"
        assert record.current_path == str(input_dir / "shot.jpg")
        assert (input_dir / "shot.jpg").exists()
        assert (raw_originals_dir / "shot.cr2").exists()
        assert not raw_path.exists()
    finally:
        manifest.close()


def test_conversion_failure_marks_manifest_failed(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    raw_path = input_dir / "broken.cr2"
    raw_path.write_bytes(b"not-a-real-raw-file")
    raw_originals_dir = tmp_path / "_raw_originals"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(raw_path, "hash1")
        run_convert_raw(raw_originals_dir, manifest)

        record = manifest.get(str(raw_path))
        assert record.raw_status == "failed"
        assert record.error_message
        assert raw_path.exists()  # untouched on failure
    finally:
        manifest.close()
