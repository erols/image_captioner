"""Integration test: caption stage against a live llama-server instance.

Skipped automatically if no server is reachable at
IMAGE_CAPTIONER_TEST_VLM_ENDPOINT (default http://127.0.0.1:8080). Run this
manually after building/launching llama-server (Task 2) to validate the
caption stage end-to-end against real hardware.
"""
import os
from pathlib import Path

import pytest
import requests

from image_captioner.caption import run_caption
from image_captioner.config import PipelineConfig
from image_captioner.manifest import Manifest
from tests.helpers import make_solid_image

ENDPOINT = os.environ.get("IMAGE_CAPTIONER_TEST_VLM_ENDPOINT", "http://127.0.0.1:8080")


def _server_reachable() -> bool:
    try:
        requests.get(f"{ENDPOINT}/health", timeout=2)
        return True
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _server_reachable(), reason="no live llama-server reachable")
def test_caption_stage_against_live_server(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (400, 300), (120, 80, 40))

    config = PipelineConfig(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        duplicates_dir=tmp_path / "_duplicates",
        raw_originals_dir=tmp_path / "_raw_originals",
        manifest_path=tmp_path / "manifest.sqlite3",
        vlm_endpoint=ENDPOINT,
    )
    manifest = Manifest(config.manifest_path)
    try:
        manifest.register(image_path, "hash1")
        manifest.update_stage(str(image_path), "raw", "done")

        run_caption(config, manifest)

        record = manifest.get(str(image_path))
        assert record.caption_status == "done"
        assert record.title
        assert record.caption
    finally:
        manifest.close()
