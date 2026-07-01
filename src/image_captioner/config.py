"""Pipeline configuration loaded from a TOML file."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPT = (
    "Write a long, evocative, descriptive caption for this image, capturing "
    "mood and meaning as well as literal content. Respond ONLY with JSON in "
    "this exact shape: "
    '{"title": "short title", "caption": "long descriptive caption", '
    '"tags": ["tag1", "tag2", "tag3"]}'
)


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    duplicates_dir: Path
    raw_originals_dir: Path
    manifest_path: Path
    vlm_endpoint: str = "http://127.0.0.1:8080"
    vlm_prompt: str = DEFAULT_PROMPT
    resize_max_dim: int = 1568
    resize_jpeg_quality: int = 92
    max_retries: int = 2
    phash_max_distance: int = 5

    @classmethod
    def from_toml(cls, path: Path) -> "PipelineConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        base = Path(data["base_dir"]).expanduser()
        return cls(
            input_dir=base / data.get("input_dir", "input"),
            output_dir=base / data.get("output_dir", "output"),
            duplicates_dir=base / data.get("duplicates_dir", "_duplicates"),
            raw_originals_dir=base / data.get("raw_originals_dir", "_raw_originals"),
            manifest_path=base / data.get("manifest_path", "manifest.sqlite3"),
            vlm_endpoint=data.get("vlm_endpoint", "http://127.0.0.1:8080"),
            vlm_prompt=data.get("vlm_prompt", DEFAULT_PROMPT),
            resize_max_dim=data.get("resize_max_dim", 1568),
            resize_jpeg_quality=data.get("resize_jpeg_quality", 92),
            max_retries=data.get("max_retries", 2),
            phash_max_distance=data.get("phash_max_distance", 5),
        )
