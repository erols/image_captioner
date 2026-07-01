"""Captioning stage: calls the local VLM for each pending, raw-converted image."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import requests

from image_captioner.config import PipelineConfig
from image_captioner.manifest import Manifest
from image_captioner.resize import resize_for_vlm
from image_captioner.vlm_client import VLMResponseError, request_caption

logger = logging.getLogger(__name__)


def run_caption(config: PipelineConfig, manifest: Manifest) -> None:
    for record in manifest.pending("caption"):
        if record.raw_status != "done":
            continue  # not yet converted from RAW (or still pending/failed)

        image_path = Path(record.current_path)
        last_error: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    resized_path = Path(tmp_dir) / "resized.jpg"
                    resize_for_vlm(
                        image_path,
                        resized_path,
                        max_dim=config.resize_max_dim,
                        quality=config.resize_jpeg_quality,
                    )
                    result = request_caption(
                        config.vlm_endpoint, resized_path, config.vlm_prompt
                    )
                manifest.update_stage(
                    record.original_path,
                    "caption",
                    "done",
                    title=result.title,
                    caption=result.caption,
                    tags=result.tags,
                )
                last_error = None
                break
            except (VLMResponseError, OSError, requests.RequestException) as exc:
                last_error = exc
                logger.warning(
                    "caption attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    config.max_retries + 1,
                    image_path,
                    exc,
                )

        if last_error is not None:
            manifest.update_stage(
                record.original_path, "caption", "failed", error_message=str(last_error)
            )
