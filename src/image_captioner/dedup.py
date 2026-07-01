"""Duplicate detection and archival stage."""
from __future__ import annotations

import shutil
from pathlib import Path

from image_captioner.formats import ALL_INPUT_EXTENSIONS, RAW_EXTENSIONS
from image_captioner.hashing import content_hash, hamming_distance, perceptual_hash
from image_captioner.manifest import Manifest


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALL_INPUT_EXTENSIONS
    )


def group_duplicates(
    images: list[Path], phashes: dict[Path, str], max_distance: int
) -> list[list[Path]]:
    """Group images whose phash hamming distance is within max_distance.

    Each returned group is sorted so the first entry is the one to keep:
    RAW files are preferred as the keeper (highest-quality source; the
    convert-raw stage will regenerate an equivalent JPEG from it), then
    shorter/lexically-earlier filenames.
    """
    parent = {p: p for p in images}

    def find(p: Path) -> Path:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: Path, b: Path) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(images):
        for b in images[i + 1 :]:
            if hamming_distance(phashes[a], phashes[b]) <= max_distance:
                union(a, b)

    groups: dict[Path, list[Path]] = {}
    for p in images:
        groups.setdefault(find(p), []).append(p)

    def sort_key(p: Path) -> tuple[int, int, str]:
        is_raw = p.suffix.lower() in RAW_EXTENSIONS
        return (0 if is_raw else 1, len(p.name), p.name)

    return [sorted(members, key=sort_key) for members in groups.values()]


def run_dedup(
    input_dir: Path,
    duplicates_dir: Path,
    manifest: Manifest,
    max_distance: int = 5,
) -> None:
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    images = discover_images(input_dir)

    phashes: dict[Path, str] = {}
    for path in images:
        manifest.register(path, content_hash(path))
        record = manifest.get(str(path))
        if record is not None and record.dedup_status == "done":
            continue
        try:
            phashes[path] = perceptual_hash(path)
        except Exception as exc:  # corrupt/unreadable image
            manifest.update_stage(str(path), "dedup", "failed", error_message=str(exc))

    groups = group_duplicates(list(phashes.keys()), phashes, max_distance)
    for group in groups:
        keeper, *dupes = group
        manifest.update_stage(
            str(keeper), "dedup", "done", phash=phashes[keeper], current_path=str(keeper)
        )
        for dup in dupes:
            dest = duplicates_dir / dup.name
            shutil.move(str(dup), dest)
            manifest.update_stage(
                str(dup), "dedup", "done", phash=phashes[dup], current_path=str(dest)
            )
            # A duplicate is archived, not processed further.
            for stage in ("raw", "caption", "publish"):
                manifest.update_stage(str(dup), stage, "skipped")
