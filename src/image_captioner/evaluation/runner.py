"""Caption-phase orchestration: run every candidate model against every image."""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

import requests

from image_captioner.evaluation.config import EvalConfig
from image_captioner.evaluation.server_manager import (
    ServerStartupError,
    start_server,
    stop_server,
    wait_for_health,
)
from image_captioner.formats import IMAGE_EXTENSIONS
from image_captioner.resize import resize_for_vlm
from image_captioner.vlm_client import VLMResponseError, request_caption

logger = logging.getLogger(__name__)


def discover_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _empty_record(error: str) -> dict:
    return {
        "status": "failed",
        "title": None,
        "caption": None,
        "tags": None,
        "elapsed_seconds": None,
        "error": error,
        "scores": None,
    }


def _caption_one(
    image_path: Path,
    endpoint: str,
    prompt: str,
    max_dim: int,
    quality: int,
    max_retries: int,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                resized_path = Path(tmp_dir) / "resized.jpg"
                resize_for_vlm(image_path, resized_path, max_dim=max_dim, quality=quality)
                result = request_caption(endpoint, resized_path, prompt)
            elapsed = time.monotonic() - start
            return {
                "status": "done",
                "title": result.title,
                "caption": result.caption,
                "tags": result.tags,
                "elapsed_seconds": elapsed,
                "error": None,
                "scores": None,
            }
        except (VLMResponseError, OSError, requests.RequestException) as exc:
            last_error = exc
            logger.warning(
                "caption attempt %d/%d failed for %s: %s",
                attempt + 1,
                max_retries + 1,
                image_path,
                exc,
            )
            if attempt < max_retries:
                time.sleep(2**attempt)
    return _empty_record(str(last_error))


def run_captioning(config: EvalConfig) -> dict:
    images = discover_images(config.image_dir)
    image_keys = [str(p.relative_to(config.image_dir)) for p in images]
    results: dict = {
        "candidates": [c.name for c in config.candidates],
        "images": image_keys,
        "results": {c.name: {} for c in config.candidates},
    }
    write_results(config.results_path, results)

    for candidate in config.candidates:
        endpoint = f"http://127.0.0.1:{config.port}"
        process = start_server(candidate, config.port)
        try:
            wait_for_health(config.port, config.server_startup_timeout)
        except ServerStartupError as exc:
            logger.warning("skipping candidate %s: %s", candidate.name, exc)
            stop_server(process)
            for key in image_keys:
                results["results"][candidate.name][key] = _empty_record(str(exc))
            write_results(config.results_path, results)
            continue

        try:
            for image_path, key in zip(images, image_keys):
                record = _caption_one(
                    image_path,
                    endpoint,
                    config.vlm_prompt,
                    config.resize_max_dim,
                    config.resize_jpeg_quality,
                    config.max_retries,
                )
                results["results"][candidate.name][key] = record
                write_results(config.results_path, results)
        finally:
            stop_server(process)

    return results
