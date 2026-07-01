"""RAW-to-JPEG conversion stage."""
from __future__ import annotations

import shutil
from pathlib import Path

import rawpy
from PIL import Image

from image_captioner.formats import RAW_EXTENSIONS
from image_captioner.manifest import Manifest


def convert_raw_to_jpeg(raw_path: Path, dest_path: Path, quality: int = 95) -> None:
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess()
    Image.fromarray(rgb).save(dest_path, format="JPEG", quality=quality)


def run_convert_raw(raw_originals_dir: Path, manifest: Manifest) -> None:
    raw_originals_dir.mkdir(parents=True, exist_ok=True)

    raw_pending = [
        r
        for r in manifest.pending("raw")
        if Path(r.current_path).suffix.lower() in RAW_EXTENSIONS
    ]
    for record in raw_pending:
        raw_path = Path(record.current_path)
        jpeg_path = raw_path.with_suffix(".jpg")
        try:
            convert_raw_to_jpeg(raw_path, jpeg_path)
        except Exception as exc:
            manifest.update_stage(record.original_path, "raw", "failed", error_message=str(exc))
            continue
        archive_path = raw_originals_dir / raw_path.name
        shutil.move(str(raw_path), archive_path)
        manifest.update_stage(record.original_path, "raw", "done", current_path=str(jpeg_path))

    # Non-RAW images simply pass through this stage.
    for record in manifest.pending("raw"):
        if Path(record.current_path).suffix.lower() not in RAW_EXTENSIONS:
            manifest.update_stage(record.original_path, "raw", "done")
