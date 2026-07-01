from pathlib import Path
from unittest.mock import patch

from image_captioner.caption import run_caption
from image_captioner.config import PipelineConfig
from image_captioner.manifest import Manifest
from image_captioner.vlm_client import CaptionResult, VLMResponseError
from tests.helpers import make_solid_image


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        duplicates_dir=tmp_path / "_duplicates",
        raw_originals_dir=tmp_path / "_raw_originals",
        manifest_path=tmp_path / "manifest.sqlite3",
        max_retries=2,
    )


def test_successful_caption_updates_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))

    manifest = Manifest(config.manifest_path)
    try:
        manifest.register(image_path, "hash1")
        manifest.update_stage(str(image_path), "raw", "done")

        fake_result = CaptionResult(title="T", caption="C", tags=["x"])
        with patch("image_captioner.caption.request_caption", return_value=fake_result):
            run_caption(config, manifest)

        record = manifest.get(str(image_path))
        assert record.caption_status == "done"
        assert record.title == "T"
        assert record.caption == "C"
        assert record.tags == ["x"]
    finally:
        manifest.close()


def test_caption_skips_images_not_yet_raw_converted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    image_path = tmp_path / "shot.cr2"
    image_path.write_bytes(b"fake-raw")

    manifest = Manifest(config.manifest_path)
    try:
        manifest.register(image_path, "hash1")  # raw_status stays 'pending'

        with patch("image_captioner.caption.request_caption") as mock_request:
            run_caption(config, manifest)
            mock_request.assert_not_called()

        record = manifest.get(str(image_path))
        assert record.caption_status == "pending"
    finally:
        manifest.close()


@patch("image_captioner.caption.time.sleep")
def test_caption_retries_then_marks_failed(mock_sleep, tmp_path: Path) -> None:
    config = _config(tmp_path)
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))

    manifest = Manifest(config.manifest_path)
    try:
        manifest.register(image_path, "hash1")
        manifest.update_stage(str(image_path), "raw", "done")

        with patch(
            "image_captioner.caption.request_caption",
            side_effect=VLMResponseError("bad json"),
        ) as mock_request:
            run_caption(config, manifest)
            assert mock_request.call_count == config.max_retries + 1

        record = manifest.get(str(image_path))
        assert record.caption_status == "failed"
        assert "bad json" in record.error_message
    finally:
        manifest.close()


@patch("image_captioner.caption.time.sleep")
def test_caption_retries_previously_failed_record_on_rerun(mock_sleep, tmp_path: Path) -> None:
    """A record marked 'failed' in an earlier run must be retried (not skipped) on a later run."""
    config = _config(tmp_path)
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))

    manifest = Manifest(config.manifest_path)
    try:
        manifest.register(image_path, "hash1")
        manifest.update_stage(str(image_path), "raw", "done")
        manifest.update_stage(
            str(image_path), "caption", "failed", error_message="previous run: bad json"
        )

        record = manifest.get(str(image_path))
        assert record.caption_status == "failed"

        fake_result = CaptionResult(title="Retried", caption="Now works", tags=["ok"])
        with patch(
            "image_captioner.caption.request_caption", return_value=fake_result
        ) as mock_request:
            run_caption(config, manifest)
            mock_request.assert_called_once()

        record = manifest.get(str(image_path))
        assert record.caption_status == "done"
        assert record.title == "Retried"
        assert record.caption == "Now works"
    finally:
        manifest.close()


@patch("image_captioner.caption.time.sleep")
def test_caption_succeeds_after_one_retry(mock_sleep, tmp_path: Path) -> None:
    config = _config(tmp_path)
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))

    manifest = Manifest(config.manifest_path)
    try:
        manifest.register(image_path, "hash1")
        manifest.update_stage(str(image_path), "raw", "done")

        fake_result = CaptionResult(title="T", caption="C", tags=[])
        with patch(
            "image_captioner.caption.request_caption",
            side_effect=[VLMResponseError("transient"), fake_result],
        ):
            run_caption(config, manifest)

        record = manifest.get(str(image_path))
        assert record.caption_status == "done"
        assert record.title == "T"
    finally:
        manifest.close()
