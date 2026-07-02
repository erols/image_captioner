"""Evaluation-harness configuration loaded from a TOML file."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from image_captioner.config import DEFAULT_PROMPT


@dataclass
class Candidate:
    name: str
    model_path: Path
    mmproj_path: Path | None = None


@dataclass
class EvalConfig:
    candidates: list[Candidate]
    image_dir: Path
    judge_model: str
    output_report: Path = Path("eval-report.md")
    results_path: Path = Path("eval-results.json")
    port: int = 8090
    vlm_prompt: str = DEFAULT_PROMPT
    resize_max_dim: int = 1568
    resize_jpeg_quality: int = 92
    max_retries: int = 2
    server_startup_timeout: float = 120.0

    @classmethod
    def from_toml(cls, path: Path) -> "EvalConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        candidates = [
            Candidate(
                name=c["name"],
                model_path=Path(c["model_path"]).expanduser(),
                mmproj_path=(
                    Path(c["mmproj_path"]).expanduser() if c.get("mmproj_path") else None
                ),
            )
            for c in data["candidates"]
        ]
        return cls(
            candidates=candidates,
            image_dir=Path(data["image_dir"]).expanduser(),
            judge_model=data["judge_model"],
            output_report=Path(data.get("output_report", "eval-report.md")),
            results_path=Path(data.get("results_path", "eval-results.json")),
            port=data.get("port", 8090),
            vlm_prompt=data.get("vlm_prompt", DEFAULT_PROMPT),
            resize_max_dim=data.get("resize_max_dim", 1568),
            resize_jpeg_quality=data.get("resize_jpeg_quality", 92),
            max_retries=data.get("max_retries", 2),
            server_startup_timeout=data.get("server_startup_timeout", 120.0),
        )
