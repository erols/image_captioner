"""Publish stage: rename images and write OKF markdown into the flat output bundle."""
from __future__ import annotations

import shutil
from pathlib import Path

from image_captioner.manifest import Manifest
from image_captioner.okf import build_okf_document
from image_captioner.slug import build_filename_stem


def run_publish(output_dir: Path, manifest: Manifest) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in manifest.pending("publish"):
        if record.caption_status != "done":
            continue  # not yet captioned

        src_path = Path(record.current_path)
        stem = build_filename_stem(record.title or "untitled", record.content_hash)
        image_dest = output_dir / f"{stem}{src_path.suffix.lower()}"
        md_dest = output_dir / f"{stem}.md"

        try:
            doc = build_okf_document(
                title=record.title or "Untitled",
                caption=record.caption or "",
                tags=record.tags,
                image_relative_path=image_dest.name,
            )
            md_dest.write_text(doc, encoding="utf-8")
            shutil.move(str(src_path), image_dest)
        except OSError as exc:
            manifest.update_stage(
                record.original_path, "publish", "failed", error_message=str(exc)
            )
            continue

        manifest.update_stage(
            record.original_path, "publish", "done", current_path=str(image_dest)
        )
