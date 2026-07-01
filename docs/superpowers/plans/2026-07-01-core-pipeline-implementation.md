# Core Image Captioning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI pipeline that takes a folder of images (including RAW), dedupes, converts RAW to JPEG, captions each image via a local VLM, and publishes a flat OKF-format markdown bundle (renamed image + `.md` sidecar) ready for LLM-driven retrieval.

**Architecture:** A `click`-based CLI with one subcommand per stage (`dedup`, `convert-raw`, `caption`, `publish`, plus a `run` convenience command), all reading/writing a shared SQLite state manifest so every stage is idempotent and resumable. The VLM is served locally by `llama-server` (llama.cpp, built with the ROCm/HIP backend) and called over its OpenAI-compatible HTTP+vision API.

**Tech Stack:** Python 3.13, `uv` (dependency/venv management), `click` (CLI), `Pillow` + `imagehash` (perceptual dedup), `rawpy` (RAW conversion), `requests` (HTTP), `PyYAML` (OKF frontmatter), `sqlite3` (stdlib, state manifest), `pytest` (tests), `llama.cpp` `llama-server` built with `-DGGML_HIP=ON`.

## Global Constraints

- Dependency/project manager: `uv` — no `pip`/`poetry`.
- Model serving: `llama-server` (llama.cpp) built with `-DGGML_HIP=ON` against ROCm 7.2.2 — never the Vulkan/RADV backend (documented mmproj correctness/crash bugs on AMD).
- VLM input resize cap: longest edge ≤ 1568px, JPEG quality 92 — one fixed cap, no per-model tuning.
- Caption call: single request per image, structured JSON response `{"title", "caption", "tags"}`.
- Caption retries: up to 2 retries with backoff, then mark `failed` in the manifest and continue the batch — never halt the run.
- Dedup: perceptual hash (phash), not byte-exact only.
- RAW conversion: `rawpy`/libraw. Supported RAW extensions: `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`. Supported standard extensions: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tif`, `.tiff`.
- Output bundle: **flat** directory, no subfolders — OKF frontmatter (`type: Image Caption`, `title`, `description`, `resource`, `tags`, `timestamp`) plus a markdown image embed and the full caption in the body.
- Filenames: slug of the title + short content-hash suffix (collision-safe without scanning existing files).
- State: SQLite manifest, one row per source image, one status column per stage (`dedup_status`, `raw_status`, `caption_status`, `publish_status`) with values `pending` / `done` / `failed` / `skipped`.
- Out of scope for this plan: multi-model evaluation (sub-project 2), tag/cross-link enrichment beyond the single VLM call (sub-project 3), any GUI.

---

## File Structure

```
pyproject.toml
src/image_captioner/
  __init__.py
  config.py          # PipelineConfig, loaded from TOML
  formats.py          # shared file-extension sets
  manifest.py          # SQLite state manifest
  hashing.py            # content hash + perceptual hash (incl. RAW thumbnails)
  dedup.py               # dedup stage
  raw_convert.py          # RAW->JPEG conversion stage
  resize.py                # resize-for-VLM utility
  vlm_client.py             # HTTP client for llama-server vision+JSON
  caption.py                 # caption stage (orchestrates resize + vlm_client + retries)
  okf.py                      # OKF frontmatter/markdown generation
  slug.py                      # filename slug + collision-hash generation
  publish.py                    # publish stage
  cli.py                         # click CLI wiring all stages together
scripts/
  build_llama_server.sh    # clone/build llama.cpp with HIP backend
  run_llama_server.sh       # launch llama-server against a model+mmproj
  smoke_test_vlm.py          # standalone de-risk script (no package dependency)
tests/
  __init__.py
  helpers.py                 # synthetic test-image generation
  test_cli.py
  test_config.py
  test_manifest.py
  test_hashing.py
  test_dedup.py
  test_raw_convert.py
  test_resize.py
  test_vlm_client.py
  test_caption.py
  test_okf.py
  test_slug.py
  test_publish.py
  test_end_to_end.py
```

---

### Task 1: Project scaffolding, config, and CLI skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/image_captioner/__init__.py`
- Create: `src/image_captioner/config.py`
- Create: `src/image_captioner/cli.py`
- Test: `tests/__init__.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `PipelineConfig` dataclass with fields `input_dir, output_dir, duplicates_dir, raw_originals_dir, manifest_path, vlm_endpoint, vlm_prompt, resize_max_dim, resize_jpeg_quality, max_retries, phash_max_distance` and classmethod `PipelineConfig.from_toml(path: Path) -> PipelineConfig`. Constant `DEFAULT_PROMPT: str`.
- Produces: `click.Group main` in `image_captioner.cli`, invocable as `image-captioner` (console script) or `uv run python -m image_captioner.cli`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "image-captioner"
version = "0.1.0"
description = "Caption, rename, and organize a folder of images using a local VLM"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "pillow>=10.0",
    "imagehash>=4.3",
    "rawpy>=0.19",
    "requests>=2.31",
    "pyyaml>=6.0",
]

[project.scripts]
image-captioner = "image_captioner.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/image_captioner"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Write `src/image_captioner/__init__.py`** (empty file marking the package)

```python
```

- [ ] **Step 3: Write `src/image_captioner/config.py`**

```python
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
```

- [ ] **Step 4: Write `src/image_captioner/cli.py`**

```python
"""Command-line entrypoint for the image-captioner pipeline."""
from __future__ import annotations

from pathlib import Path

import click

from image_captioner.config import PipelineConfig


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("pipeline.toml"),
    show_default=True,
)
@click.pass_context
def main(ctx: click.Context, config_path: Path) -> None:
    """Turn a folder of images into captioned, renamed OKF markdown notes."""
    ctx.obj = PipelineConfig.from_toml(config_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 6: Write `tests/test_config.py`**

```python
from pathlib import Path

from image_captioner.config import PipelineConfig


def test_from_toml_uses_defaults_for_optional_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(f'base_dir = "{tmp_path}"\n')

    config = PipelineConfig.from_toml(config_path)

    assert config.input_dir == tmp_path / "input"
    assert config.output_dir == tmp_path / "output"
    assert config.duplicates_dir == tmp_path / "_duplicates"
    assert config.raw_originals_dir == tmp_path / "_raw_originals"
    assert config.manifest_path == tmp_path / "manifest.sqlite3"
    assert config.resize_max_dim == 1568
    assert config.max_retries == 2


def test_from_toml_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        f'base_dir = "{tmp_path}"\n'
        'resize_max_dim = 1024\n'
        'vlm_endpoint = "http://example.local:9000"\n'
    )

    config = PipelineConfig.from_toml(config_path)

    assert config.resize_max_dim == 1024
    assert config.vlm_endpoint == "http://example.local:9000"
```

- [ ] **Step 7: Write `tests/test_cli.py`**

```python
from click.testing import CliRunner

from image_captioner.cli import main


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
```

- [ ] **Step 8: Sync dependencies and create the venv**

Run: `uv sync`
Expected: creates/updates `.venv` and `uv.lock`, installs `image-captioner` in editable mode.

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -v`
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock src/image_captioner/__init__.py src/image_captioner/config.py src/image_captioner/cli.py tests/__init__.py tests/test_config.py tests/test_cli.py
git commit -m "feat: project scaffolding, config loader, and CLI skeleton"
```

---

### Task 2: Build llama-server (ROCm/HIP) and de-risk vision+JSON captioning

**Files:**
- Create: `scripts/build_llama_server.sh`
- Create: `scripts/run_llama_server.sh`
- Create: `scripts/smoke_test_vlm.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately standalone — the package doesn't need to exist yet for this de-risk check to run).
- Produces: a working `llama-server` binary at `$LLAMA_CPP_DIR/build/bin/llama-server` (default `~/.local/src/llama.cpp/build/bin/llama-server`), and empirical confirmation of whether vision+JSON output works, which Task 8 (`vlm_client.py`) must be consistent with.

This task is a **de-risk spike**, per the design doc's flagged risk: llama.cpp's multimodal (`mtmd`) support for `mllama` (Llama-3.2-Vision) and for JoyCaption's from-scratch LLaVA architecture is not guaranteed to work out of the box, and JSON-constrained output combined with an image in the same request is unproven. Do this before investing further in the pipeline.

- [ ] **Step 1: Write `scripts/build_llama_server.sh`**

```bash
#!/usr/bin/env bash
# Clones (or updates) and builds llama.cpp's llama-server with the ROCm/HIP
# backend. Do NOT use the Vulkan/RADV backend here — it has documented
# mmproj (vision encoder) correctness and crash bugs on AMD GPUs.
set -euo pipefail

REPO_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull --ff-only

cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/bin/hipcc" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j"$(nproc)"

echo "Built: $REPO_DIR/build/bin/llama-server"
```

- [ ] **Step 2: Write `scripts/run_llama_server.sh`**

```bash
#!/usr/bin/env bash
# Launches llama-server against a given GGUF model (+ optional mmproj for
# vision). Sets the gfx1151 ROCm "Preview" override required on Strix Halo.
set -euo pipefail

MODEL_PATH="${1:?usage: run_llama_server.sh MODEL_GGUF [MMPROJ_GGUF]}"
MMPROJ_PATH="${2:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp}"
PORT="${LLAMA_SERVER_PORT:-8080}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

ARGS=(-m "$MODEL_PATH" --port "$PORT" -ngl 999)
if [ -n "$MMPROJ_PATH" ]; then
  ARGS+=(--mmproj "$MMPROJ_PATH")
fi

exec "$LLAMA_CPP_DIR/build/bin/llama-server" "${ARGS[@]}"
```

- [ ] **Step 3: Make both scripts executable**

Run: `chmod +x scripts/build_llama_server.sh scripts/run_llama_server.sh`

- [ ] **Step 4: Write `scripts/smoke_test_vlm.py`**

```python
#!/usr/bin/env python3
"""
De-risk smoke test for llama-server vision + JSON captioning.

Confirms that a running llama-server instance can:
  1. accept an image via the OpenAI-compatible vision chat API, and
  2. return content that contains a parseable {"title", "caption", "tags"}
     JSON object.

Deliberately standalone (no dependency on the image_captioner package) so it
can run the moment the server is built, before the rest of the pipeline
exists.

Usage: python3 scripts/smoke_test_vlm.py IMAGE_PATH [ENDPOINT]
"""
import base64
import json
import sys
from pathlib import Path

import requests

PROMPT = (
    "Write a long, evocative, descriptive caption for this image, capturing "
    "mood and meaning as well as literal content. Respond ONLY with JSON in "
    "this exact shape: "
    '{"title": "short title", "caption": "long descriptive caption", '
    '"tags": ["tag1", "tag2", "tag3"]}'
)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = Path(sys.argv[1])
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"

    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.7,
    }
    response = requests.post(
        f"{endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=120
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print("--- raw model output ---")
    print(content)

    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        print("\nFAIL: no JSON object found in output")
        sys.exit(1)
    data = json.loads(content[start : end + 1])
    for key in ("title", "caption", "tags"):
        assert key in data, f"missing key: {key}"
    print("\n--- parsed ---")
    print(f"Title:   {data['title']}")
    print(f"Caption: {data['caption']}")
    print(f"Tags:    {data['tags']}")
    print("\nPASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Build llama-server**

Run: `bash scripts/build_llama_server.sh`
Expected: ends with `Built: <path>/build/bin/llama-server` and the file exists and is executable.

- [ ] **Step 6: Launch the server against Qwen3-VL-8B (most likely to work first)**

Run (in a separate terminal / background process):
`bash scripts/run_llama_server.sh /mnt/models/hf_ggufs/vlm/unsloth_Qwen3-VL-8B-Instruct-BF16.gguf`

Expected: log output showing the HTTP server listening on port 8080, no crash during model/mmproj load. If Qwen3-VL needs a separate mmproj file that isn't bundled in this GGUF, note that and acquire/build the correct mmproj before proceeding — do not skip this check.

- [ ] **Step 7: Run the smoke test against a real image**

Run: `uv run python scripts/smoke_test_vlm.py /path/to/any/test/photo.jpg`
Expected: prints raw model output, then parsed Title/Caption/Tags, then `PASS`.

If this fails (no JSON in output, crash, wrong/garbled caption): try lowering `temperature`, simplify the prompt, or fall back to plain-text JSON extraction only (already what `parse_caption_json`/this script does — no server-side grammar constraint is used, so this is the realistic baseline). Record what worked; Task 8's `vlm_client.py` must match this validated approach.

- [ ] **Step 8: Repeat against JoyCaption and Llama-3.2-Vision**

Run the same server/smoke-test cycle against:
- `/mnt/models/hf_ggufs/vlm/llama-joycaption-beta-one-hf-llava.f16.gguf` (once its download finishes — this is the priority model for caption quality)
- `/mnt/models/hf_ggufs/vlm/Llama-3.2-11B-Vision-Instruct.Q8_0.gguf` (mllama architecture — most likely to have multimodal support gaps)

Expected: `PASS` for each, or a documented failure mode per model. It's acceptable for the MVP to proceed with only one working model (e.g. Qwen3-VL) as long as at least one passes — the model-agnostic config (`vlm_endpoint`, `vlm_prompt`) means the others can be revisited in the evaluation harness (sub-project 2) without pipeline changes.

- [ ] **Step 9: Commit**

```bash
git add scripts/build_llama_server.sh scripts/run_llama_server.sh scripts/smoke_test_vlm.py
git commit -m "feat: llama-server build/run scripts and vision+JSON de-risk smoke test"
```

---

### Task 3: SQLite state manifest

**Files:**
- Create: `src/image_captioner/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `STAGES: tuple[str, ...] = ("dedup", "raw", "caption", "publish")`
  - `class ImageRecord` (dataclass) with fields `original_path, current_path, content_hash, phash, dedup_status, raw_status, caption_status, publish_status, title, caption, tags: list[str], error_message, updated_at`
  - `class Manifest`:
    - `__init__(self, db_path: Path)`
    - `close(self) -> None`
    - `register(self, original_path: Path, content_hash: str) -> None`
    - `get(self, original_path: str) -> ImageRecord | None`
    - `update_stage(self, original_path: str, stage: str, status: str, *, current_path=None, phash=None, title=None, caption=None, tags=None, error_message=None) -> None`
    - `pending(self, stage: str) -> list[ImageRecord]`
    - `failed(self, stage: str) -> list[ImageRecord]`

- [ ] **Step 1: Write the failing test in `tests/test_manifest.py`**

```python
from pathlib import Path

import pytest

from image_captioner.manifest import Manifest


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    m = Manifest(tmp_path / "manifest.sqlite3")
    yield m
    m.close()


def test_register_then_get(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")

    record = manifest.get("/photos/a.jpg")

    assert record is not None
    assert record.current_path == "/photos/a.jpg"
    assert record.content_hash == "hash1"
    assert record.dedup_status == "pending"
    assert record.raw_status == "pending"
    assert record.caption_status == "pending"
    assert record.publish_status == "pending"
    assert record.tags == []


def test_register_is_idempotent(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")
    manifest.register(Path("/photos/a.jpg"), "hash1")

    manifest.update_stage("/photos/a.jpg", "dedup", "done")
    manifest.register(Path("/photos/a.jpg"), "hash1")  # must not reset status

    record = manifest.get("/photos/a.jpg")
    assert record.dedup_status == "done"


def test_update_stage_sets_status_and_fields(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")

    manifest.update_stage(
        "/photos/a.jpg",
        "caption",
        "done",
        title="A Title",
        caption="A caption.",
        tags=["mood", "outdoor"],
    )

    record = manifest.get("/photos/a.jpg")
    assert record.caption_status == "done"
    assert record.title == "A Title"
    assert record.caption == "A caption."
    assert record.tags == ["mood", "outdoor"]


def test_update_stage_rejects_unknown_stage(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")
    with pytest.raises(ValueError):
        manifest.update_stage("/photos/a.jpg", "bogus", "done")


def test_pending_and_failed_queries(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")
    manifest.register(Path("/photos/b.jpg"), "hash2")
    manifest.update_stage("/photos/a.jpg", "caption", "failed", error_message="boom")

    pending = {r.original_path for r in manifest.pending("caption")}
    failed = {r.original_path for r in manifest.failed("caption")}

    assert pending == {"/photos/b.jpg"}
    assert failed == {"/photos/a.jpg"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.manifest'`

- [ ] **Step 3: Write `src/image_captioner/manifest.py`**

```python
"""SQLite-backed state manifest tracking each image through pipeline stages."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("dedup", "raw", "caption", "publish")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    original_path TEXT PRIMARY KEY,
    current_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    phash TEXT,
    dedup_status TEXT NOT NULL DEFAULT 'pending',
    raw_status TEXT NOT NULL DEFAULT 'pending',
    caption_status TEXT NOT NULL DEFAULT 'pending',
    publish_status TEXT NOT NULL DEFAULT 'pending',
    title TEXT,
    caption TEXT,
    tags TEXT,
    error_message TEXT,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class ImageRecord:
    original_path: str
    current_path: str
    content_hash: str
    phash: str | None
    dedup_status: str
    raw_status: str
    caption_status: str
    publish_status: str
    title: str | None
    caption: str | None
    tags: list[str]
    error_message: str | None
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ImageRecord":
        return cls(
            original_path=row["original_path"],
            current_path=row["current_path"],
            content_hash=row["content_hash"],
            phash=row["phash"],
            dedup_status=row["dedup_status"],
            raw_status=row["raw_status"],
            caption_status=row["caption_status"],
            publish_status=row["publish_status"],
            title=row["title"],
            caption=row["caption"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            error_message=row["error_message"],
            updated_at=row["updated_at"],
        )


class Manifest:
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def register(self, original_path: Path, content_hash: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO images
                (original_path, current_path, content_hash, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (str(original_path), str(original_path), content_hash, now),
        )
        self._conn.commit()

    def get(self, original_path: str) -> ImageRecord | None:
        row = self._conn.execute(
            "SELECT * FROM images WHERE original_path = ?", (original_path,)
        ).fetchone()
        return ImageRecord.from_row(row) if row else None

    def update_stage(
        self,
        original_path: str,
        stage: str,
        status: str,
        *,
        current_path: str | None = None,
        phash: str | None = None,
        title: str | None = None,
        caption: str | None = None,
        tags: list[str] | None = None,
        error_message: str | None = None,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        fields: dict[str, object] = {
            f"{stage}_status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if current_path is not None:
            fields["current_path"] = current_path
        if phash is not None:
            fields["phash"] = phash
        if title is not None:
            fields["title"] = title
        if caption is not None:
            fields["caption"] = caption
        if tags is not None:
            fields["tags"] = json.dumps(tags)
        if error_message is not None:
            fields["error_message"] = error_message
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        self._conn.execute(
            f"UPDATE images SET {set_clause} WHERE original_path = ?",
            (*fields.values(), original_path),
        )
        self._conn.commit()

    def pending(self, stage: str) -> list[ImageRecord]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        rows = self._conn.execute(
            f"SELECT * FROM images WHERE {stage}_status = 'pending'"
        ).fetchall()
        return [ImageRecord.from_row(r) for r in rows]

    def failed(self, stage: str) -> list[ImageRecord]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        rows = self._conn.execute(
            f"SELECT * FROM images WHERE {stage}_status = 'failed'"
        ).fetchall()
        return [ImageRecord.from_row(r) for r in rows]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/manifest.py tests/test_manifest.py
git commit -m "feat: SQLite state manifest for pipeline stage tracking"
```

---

### Task 4: Hashing utilities (content hash + perceptual hash, including RAW thumbnails)

**Files:**
- Create: `src/image_captioner/formats.py`
- Create: `src/image_captioner/hashing.py`
- Create: `tests/helpers.py`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `formats.py`: `IMAGE_EXTENSIONS: set[str]`, `RAW_EXTENSIONS: set[str]`, `ALL_INPUT_EXTENSIONS: set[str]`
  - `hashing.py`: `content_hash(path: Path) -> str`, `perceptual_hash(path: Path) -> str`, `hamming_distance(hash_a: str, hash_b: str) -> int`
  - `tests/helpers.py`: `make_solid_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None` (reused by later test tasks)

- [ ] **Step 1: Write `src/image_captioner/formats.py`**

```python
"""Shared file-format extension sets."""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"}
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng"}
ALL_INPUT_EXTENSIONS = IMAGE_EXTENSIONS | RAW_EXTENSIONS
```

- [ ] **Step 2: Write `tests/helpers.py`**

```python
"""Shared test helpers for synthesizing sample images."""
from pathlib import Path

from PIL import Image


def make_solid_image(
    path: Path, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    img = Image.new("RGB", size, color)
    img.save(path)
```

- [ ] **Step 3: Write the failing test in `tests/test_hashing.py`**

```python
from pathlib import Path

from PIL import Image

from image_captioner.hashing import content_hash, hamming_distance, perceptual_hash
from tests.helpers import make_solid_image


def test_content_hash_deterministic_and_sensitive_to_bytes(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    make_solid_image(path_a, (100, 100), (255, 0, 0))
    make_solid_image(path_b, (100, 100), (0, 255, 0))

    assert content_hash(path_a) == content_hash(path_a)
    assert content_hash(path_a) != content_hash(path_b)


def test_perceptual_hash_similar_for_resized_recompressed_copy(tmp_path: Path) -> None:
    original = tmp_path / "original.jpg"
    make_solid_image(original, (800, 600), (30, 60, 90))

    resized = tmp_path / "resized.jpg"
    with Image.open(original) as img:
        img.resize((200, 150)).save(resized, quality=80)

    distant = tmp_path / "distant.jpg"
    make_solid_image(distant, (800, 600), (220, 20, 140))

    near_distance = hamming_distance(perceptual_hash(original), perceptual_hash(resized))
    far_distance = hamming_distance(perceptual_hash(original), perceptual_hash(distant))

    assert near_distance <= 5
    assert far_distance > near_distance
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.hashing'`

- [ ] **Step 5: Write `src/image_captioner/hashing.py`**

```python
"""Content and perceptual hashing utilities."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import imagehash
import rawpy
from PIL import Image

from image_captioner.formats import RAW_EXTENSIONS


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_for_hash(path: Path) -> Image.Image:
    if path.suffix.lower() in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(thumb.data))
        return Image.fromarray(thumb.data)
    return Image.open(path)


def perceptual_hash(path: Path) -> str:
    with _load_for_hash(path) as img:
        return str(imagehash.phash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/image_captioner/formats.py src/image_captioner/hashing.py tests/helpers.py tests/test_hashing.py
git commit -m "feat: content/perceptual hashing utilities with RAW thumbnail support"
```

---

### Task 5: Dedup stage and CLI subcommand

**Files:**
- Create: `src/image_captioner/dedup.py`
- Modify: `src/image_captioner/cli.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `formats.ALL_INPUT_EXTENSIONS`, `formats.RAW_EXTENSIONS`; `hashing.content_hash`, `hashing.perceptual_hash`, `hashing.hamming_distance`; `manifest.Manifest` (`register`, `get`, `update_stage`); `config.PipelineConfig`.
- Produces: `discover_images(input_dir: Path) -> list[Path]`; `group_duplicates(images: list[Path], phashes: dict[Path, str], max_distance: int) -> list[list[Path]]` (pure, first element of each group is the keeper); `run_dedup(input_dir: Path, duplicates_dir: Path, manifest: Manifest, max_distance: int = 5) -> None`. Adds `dedup` command to the CLI group.

- [ ] **Step 1: Write the failing test in `tests/test_dedup.py`**

```python
from pathlib import Path

from image_captioner.dedup import group_duplicates, run_dedup
from image_captioner.manifest import Manifest
from tests.helpers import make_solid_image


def test_group_duplicates_groups_within_threshold_keeping_shortest_name() -> None:
    a = Path("photo.jpg")
    a_copy = Path("photo_copy.jpg")
    b = Path("other.jpg")
    phashes = {a: "0" * 16, a_copy: "1" * 16, b: "f" * 16}
    # a and a_copy differ in every bit vs each other in this synthetic hash,
    # so instead assert on hamming_distance-driven grouping using real values:
    phashes = {a: "ffff000000000000", a_copy: "ffff000000000001", b: "0000ffffffffffff"}

    groups = group_duplicates([a, a_copy, b], phashes, max_distance=2)

    groups_as_sets = [set(g) for g in groups]
    assert {a, a_copy} in groups_as_sets
    assert [b] in [list(g) for g in groups if len(g) == 1]
    dup_group = next(g for g in groups if len(g) == 2)
    assert dup_group[0] == a  # shorter name kept first


def test_group_duplicates_prefers_raw_as_keeper() -> None:
    raw_file = Path("shot.cr2")
    jpeg_file = Path("shot.jpg")
    phashes = {raw_file: "0" * 16, jpeg_file: "0" * 16}

    groups = group_duplicates([raw_file, jpeg_file], phashes, max_distance=2)

    assert groups[0][0] == raw_file


def test_run_dedup_moves_duplicates_and_updates_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    duplicates_dir = tmp_path / "_duplicates"
    make_solid_image(input_dir / "sunset.jpg", (800, 600), (200, 100, 50))
    make_solid_image(input_dir / "sunset_dup.jpg", (800, 600), (200, 100, 50))
    make_solid_image(input_dir / "forest.jpg", (800, 600), (10, 90, 20))

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        run_dedup(input_dir, duplicates_dir, manifest, max_distance=5)

        remaining = sorted(p.name for p in input_dir.iterdir())
        moved = sorted(p.name for p in duplicates_dir.iterdir())
        assert len(remaining) == 2  # one of the near-duplicates moved out
        assert len(moved) == 1

        moved_record = manifest.get(str(input_dir / moved[0]))
        assert moved_record.dedup_status == "done"
        assert moved_record.raw_status == "skipped"
        assert moved_record.caption_status == "skipped"
        assert moved_record.publish_status == "skipped"
    finally:
        manifest.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.dedup'`

- [ ] **Step 3: Write `src/image_captioner/dedup.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dedup.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Wire the `dedup` subcommand into the CLI**

Modify `src/image_captioner/cli.py`, adding these imports and command after the `main` group definition:

```python
from image_captioner.dedup import run_dedup
from image_captioner.manifest import Manifest


@main.command()
@click.pass_obj
def dedup(config: PipelineConfig) -> None:
    """Find and archive near-duplicate images."""
    manifest = Manifest(config.manifest_path)
    try:
        run_dedup(config.input_dir, config.duplicates_dir, manifest, config.phash_max_distance)
    finally:
        manifest.close()
```

- [ ] **Step 6: Run the full test suite to verify nothing broke**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/image_captioner/dedup.py src/image_captioner/cli.py tests/test_dedup.py
git commit -m "feat: perceptual-hash dedup stage and CLI subcommand"
```

---

### Task 6: RAW conversion stage and CLI subcommand

**Files:**
- Create: `src/image_captioner/raw_convert.py`
- Modify: `src/image_captioner/cli.py`
- Test: `tests/test_raw_convert.py`

**Interfaces:**
- Consumes: `formats.RAW_EXTENSIONS`; `manifest.Manifest` (`pending`, `update_stage`); `config.PipelineConfig`.
- Produces: `convert_raw_to_jpeg(raw_path: Path, dest_path: Path, quality: int = 95) -> None`; `run_convert_raw(raw_originals_dir: Path, manifest: Manifest) -> None`. Adds `convert-raw` command to the CLI group.

- [ ] **Step 1: Write the failing test in `tests/test_raw_convert.py`**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_raw_convert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.raw_convert'`

- [ ] **Step 3: Write `src/image_captioner/raw_convert.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_raw_convert.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Wire the `convert-raw` subcommand into the CLI**

Modify `src/image_captioner/cli.py`, adding:

```python
from image_captioner.raw_convert import run_convert_raw


@main.command(name="convert-raw")
@click.pass_obj
def convert_raw_cmd(config: PipelineConfig) -> None:
    """Convert RAW originals to JPEG and archive the originals."""
    manifest = Manifest(config.manifest_path)
    try:
        run_convert_raw(config.raw_originals_dir, manifest)
    finally:
        manifest.close()
```

- [ ] **Step 6: Run the full test suite to verify nothing broke**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/image_captioner/raw_convert.py src/image_captioner/cli.py tests/test_raw_convert.py
git commit -m "feat: RAW-to-JPEG conversion stage and CLI subcommand"
```

---

### Task 7: Resize-for-VLM utility

**Files:**
- Create: `src/image_captioner/resize.py`
- Test: `tests/test_resize.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resize_for_vlm(src_path: Path, dest_path: Path, max_dim: int = 1568, quality: int = 92) -> None`

- [ ] **Step 1: Write the failing test in `tests/test_resize.py`**

```python
from pathlib import Path

from PIL import Image

from image_captioner.resize import resize_for_vlm
from tests.helpers import make_solid_image


def test_large_image_is_downscaled_to_max_dim(tmp_path: Path) -> None:
    src = tmp_path / "large.jpg"
    dest = tmp_path / "resized.jpg"
    make_solid_image(src, (3000, 2000), (100, 150, 200))

    resize_for_vlm(src, dest, max_dim=1568, quality=92)

    with Image.open(dest) as img:
        assert max(img.size) <= 1568
        assert img.size[0] / img.size[1] == 3000 / 2000


def test_small_image_is_not_upscaled(tmp_path: Path) -> None:
    src = tmp_path / "small.jpg"
    dest = tmp_path / "resized.jpg"
    make_solid_image(src, (200, 100), (10, 10, 10))

    resize_for_vlm(src, dest, max_dim=1568, quality=92)

    with Image.open(dest) as img:
        assert img.size == (200, 100)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_resize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.resize'`

- [ ] **Step 3: Write `src/image_captioner/resize.py`**

```python
"""Resize images for VLM input."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def resize_for_vlm(
    src_path: Path, dest_path: Path, max_dim: int = 1568, quality: int = 92
) -> None:
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, max_dim / max(width, height))
        if scale < 1.0:
            img = img.resize(
                (round(width * scale), round(height * scale)), Image.LANCZOS
            )
        img.save(dest_path, format="JPEG", quality=quality)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_resize.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/resize.py tests/test_resize.py
git commit -m "feat: fixed-cap resize utility for VLM input"
```

---

### Task 8: VLM HTTP client

**Files:**
- Create: `src/image_captioner/vlm_client.py`
- Test: `tests/test_vlm_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Prompt/parsing approach must match whatever Task 2's smoke test validated (plain-text JSON extraction, no server-side grammar constraint assumed).
- Produces: `class VLMResponseError(Exception)`; `@dataclass class CaptionResult` with fields `title: str, caption: str, tags: list[str]`; `parse_caption_json(content: str) -> CaptionResult`; `request_caption(endpoint: str, image_path: Path, prompt: str, timeout: float = 120.0) -> CaptionResult`.

- [ ] **Step 1: Write the failing test in `tests/test_vlm_client.py`**

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_captioner.vlm_client import (
    CaptionResult,
    VLMResponseError,
    parse_caption_json,
    request_caption,
)


def test_parse_caption_json_clean() -> None:
    content = '{"title": "Quiet Dock", "caption": "A quiet dock at dawn.", "tags": ["calm", "water"]}'
    result = parse_caption_json(content)
    assert result == CaptionResult(
        title="Quiet Dock", caption="A quiet dock at dawn.", tags=["calm", "water"]
    )


def test_parse_caption_json_embedded_in_prose() -> None:
    content = (
        "Sure, here's the caption:\n```json\n"
        '{"title": "Quiet Dock", "caption": "A quiet dock.", "tags": ["calm"]}'
        "\n```"
    )
    result = parse_caption_json(content)
    assert result.title == "Quiet Dock"
    assert result.tags == ["calm"]


def test_parse_caption_json_missing_key_raises() -> None:
    with pytest.raises(VLMResponseError):
        parse_caption_json('{"title": "X", "caption": "Y"}')


def test_parse_caption_json_malformed_raises() -> None:
    with pytest.raises(VLMResponseError):
        parse_caption_json("not json at all, no braces")


def test_request_caption_posts_expected_payload_and_parses_response(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"title": "T", "caption": "C", "tags": ["a", "b"]}'
                }
            }
        ]
    }

    with patch("image_captioner.vlm_client.requests.post", return_value=fake_response) as post:
        result = request_caption("http://127.0.0.1:8080", image_path, "describe this")

    assert result == CaptionResult(title="T", caption="C", tags=["a", "b"])
    called_url = post.call_args.args[0]
    assert called_url == "http://127.0.0.1:8080/v1/chat/completions"
    payload = post.call_args.kwargs["json"]
    content_parts = payload["messages"][0]["content"]
    assert content_parts[0]["text"] == "describe this"
    assert content_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_vlm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.vlm_client'`

- [ ] **Step 3: Write `src/image_captioner/vlm_client.py`**

```python
"""HTTP client for the local llama-server vision+JSON captioning endpoint."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import requests


class VLMResponseError(Exception):
    """Raised when the VLM response can't be parsed into the expected shape."""


@dataclass
class CaptionResult:
    title: str
    caption: str
    tags: list[str]


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def parse_caption_json(content: str) -> CaptionResult:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise VLMResponseError(f"no JSON object found in content: {content!r}")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VLMResponseError(f"invalid JSON in content: {content!r}") from exc
    for key in ("title", "caption", "tags"):
        if key not in data:
            raise VLMResponseError(f"missing key {key!r} in VLM JSON: {data!r}")
    tags = data["tags"]
    if not isinstance(tags, list):
        raise VLMResponseError(f"'tags' must be a list, got: {tags!r}")
    return CaptionResult(
        title=str(data["title"]), caption=str(data["caption"]), tags=[str(t) for t in tags]
    )


def request_caption(
    endpoint: str, image_path: Path, prompt: str, timeout: float = 120.0
) -> CaptionResult:
    b64 = _encode_image(image_path)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.7,
    }
    response = requests.post(
        f"{endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout
    )
    response.raise_for_status()
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise VLMResponseError(f"unexpected response shape: {body}") from exc
    return parse_caption_json(content)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_vlm_client.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/vlm_client.py tests/test_vlm_client.py
git commit -m "feat: llama-server HTTP client with lenient JSON caption parsing"
```

---

### Task 9: Caption stage and CLI subcommand

**Files:**
- Create: `src/image_captioner/caption.py`
- Modify: `src/image_captioner/cli.py`
- Test: `tests/test_caption.py`
- Test: `tests/test_caption_integration.py`

**Interfaces:**
- Consumes: `config.PipelineConfig`; `manifest.Manifest` (`pending`, `update_stage`); `resize.resize_for_vlm`; `vlm_client.request_caption`, `vlm_client.VLMResponseError`.
- Produces: `run_caption(config: PipelineConfig, manifest: Manifest) -> None`. Adds `caption` command to the CLI group.

- [ ] **Step 1: Write the failing test in `tests/test_caption.py`**

```python
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


def test_caption_retries_then_marks_failed(tmp_path: Path) -> None:
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


def test_caption_succeeds_after_one_retry(tmp_path: Path) -> None:
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_caption.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.caption'`

- [ ] **Step 3: Write `src/image_captioner/caption.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_caption.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Wire the `caption` subcommand into the CLI**

Modify `src/image_captioner/cli.py`, adding:

```python
from image_captioner.caption import run_caption


@main.command()
@click.pass_obj
def caption(config: PipelineConfig) -> None:
    """Caption each pending, raw-converted image via the local VLM."""
    manifest = Manifest(config.manifest_path)
    try:
        run_caption(config, manifest)
    finally:
        manifest.close()
```

- [ ] **Step 6: Write the live-server integration test in `tests/test_caption_integration.py`**

This test is skipped automatically unless a real `llama-server` (built in Task 2) is running and reachable — it exists to validate the full caption stage against real hardware, not to run in CI.

```python
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
```

- [ ] **Step 7: Run it to confirm the skip behavior works without a server**

Run: `uv run pytest tests/test_caption_integration.py -v`
Expected: `SKIPPED (no live llama-server reachable)` — this is correct/expected when no server is running. If you have `llama-server` running locally from Task 2, this should instead run and PASS.

- [ ] **Step 8: Run the full test suite to verify nothing broke**

Run: `uv run pytest -v`
Expected: all tests PASS (or SKIPPED for the integration test, per above).

- [ ] **Step 9: Commit**

```bash
git add src/image_captioner/caption.py src/image_captioner/cli.py tests/test_caption.py tests/test_caption_integration.py
git commit -m "feat: caption stage with retry/failure handling, CLI subcommand, and live-server integration test"
```

---

### Task 10: OKF frontmatter/markdown generation

**Files:**
- Create: `src/image_captioner/okf.py`
- Test: `tests/test_okf.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_okf_document(title: str, caption: str, tags: list[str], image_relative_path: str, timestamp: datetime | None = None) -> str`

- [ ] **Step 1: Write the failing test in `tests/test_okf.py`**

```python
from datetime import datetime, timezone

import yaml

from image_captioner.okf import build_okf_document


def test_build_okf_document_has_valid_frontmatter_and_body() -> None:
    doc = build_okf_document(
        title="Quiet Harbor at Dusk",
        caption="A quiet harbor at dusk. Boats rest at anchor under a fading sky.",
        tags=["calm", "harbor", "dusk"],
        image_relative_path="quiet-harbor-at-dusk-a1b2c3.jpg",
        timestamp=datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert doc.startswith("---\n")
    _, frontmatter_raw, body = doc.split("---\n", 2)
    frontmatter = yaml.safe_load(frontmatter_raw)

    assert frontmatter["type"] == "Image Caption"
    assert frontmatter["title"] == "Quiet Harbor at Dusk"
    assert frontmatter["description"] == "A quiet harbor at dusk."
    assert frontmatter["resource"] == "quiet-harbor-at-dusk-a1b2c3.jpg"
    assert frontmatter["tags"] == ["calm", "harbor", "dusk"]
    assert frontmatter["timestamp"] == "2026-07-01T12:00:00+00:00"

    assert "![Quiet Harbor at Dusk](quiet-harbor-at-dusk-a1b2c3.jpg)" in body
    assert "Boats rest at anchor under a fading sky." in body
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_okf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.okf'`

- [ ] **Step 3: Write `src/image_captioner/okf.py`**

```python
"""OKF-format frontmatter and markdown document generation."""
from __future__ import annotations

from datetime import datetime, timezone

import yaml


def build_okf_document(
    title: str,
    caption: str,
    tags: list[str],
    image_relative_path: str,
    timestamp: datetime | None = None,
) -> str:
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    first_sentence = caption.split(". ")[0].rstrip(".") + "."
    frontmatter = {
        "type": "Image Caption",
        "title": title,
        "description": first_sentence,
        "resource": image_relative_path,
        "tags": tags,
        "timestamp": ts,
    }
    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    body = f"![{title}]({image_relative_path})\n\n{caption}\n"
    return f"---\n{yaml_block}---\n\n{body}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_okf.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/okf.py tests/test_okf.py
git commit -m "feat: OKF frontmatter/markdown document generation"
```

---

### Task 11: Filename slug + collision-hash generation

**Files:**
- Create: `src/image_captioner/slug.py`
- Test: `tests/test_slug.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `slugify(title: str) -> str`; `build_filename_stem(title: str, content_hash: str, hash_len: int = 6) -> str`

- [ ] **Step 1: Write the failing test in `tests/test_slug.py`**

```python
from image_captioner.slug import build_filename_stem, slugify


def test_slugify_basic() -> None:
    assert slugify("Quiet Harbor at Dusk") == "quiet-harbor-at-dusk"


def test_slugify_strips_punctuation_and_collapses_hyphens() -> None:
    assert slugify("Wow!! It's -- a Sunset??") == "wow-it-s-a-sunset"


def test_slugify_empty_title_falls_back() -> None:
    assert slugify("   ") == "untitled"


def test_build_filename_stem_includes_hash_suffix() -> None:
    stem = build_filename_stem("Quiet Harbor", "abcdef1234567890", hash_len=6)
    assert stem == "quiet-harbor-abcdef"


def test_build_filename_stem_differs_for_different_hashes_with_same_title() -> None:
    stem_a = build_filename_stem("Quiet Harbor", "111111", hash_len=6)
    stem_b = build_filename_stem("Quiet Harbor", "222222", hash_len=6)
    assert stem_a != stem_b
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_slug.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.slug'`

- [ ] **Step 3: Write `src/image_captioner/slug.py`**

```python
"""Filename slug generation with collision-safe hashing."""
from __future__ import annotations

import re


def slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "untitled"


def build_filename_stem(title: str, content_hash: str, hash_len: int = 6) -> str:
    return f"{slugify(title)}-{content_hash[:hash_len]}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_slug.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/slug.py tests/test_slug.py
git commit -m "feat: collision-safe filename slug generation"
```

---

### Task 12: Publish stage and CLI subcommand

**Files:**
- Create: `src/image_captioner/publish.py`
- Modify: `src/image_captioner/cli.py`
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: `manifest.Manifest` (`pending`, `update_stage`); `okf.build_okf_document`; `slug.build_filename_stem`.
- Produces: `run_publish(output_dir: Path, manifest: Manifest) -> None`. Adds `publish` command to the CLI group.

- [ ] **Step 1: Write the failing test in `tests/test_publish.py`**

```python
from pathlib import Path

from image_captioner.manifest import Manifest
from image_captioner.publish import run_publish
from tests.helpers import make_solid_image


def test_publish_renames_image_and_writes_okf_markdown(tmp_path: Path) -> None:
    image_path = tmp_path / "IMG_0001.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))
    output_dir = tmp_path / "output"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(image_path, "abcdef123456")
        manifest.update_stage(str(image_path), "raw", "done")
        manifest.update_stage(
            str(image_path),
            "caption",
            "done",
            title="Quiet Harbor",
            caption="A quiet harbor at dusk.",
            tags=["calm", "harbor"],
        )

        run_publish(output_dir, manifest)

        jpg_files = list(output_dir.glob("*.jpg"))
        md_files = list(output_dir.glob("*.md"))
        assert len(jpg_files) == 1
        assert len(md_files) == 1
        assert jpg_files[0].stem == md_files[0].stem
        assert jpg_files[0].stem == "quiet-harbor-abcdef"

        content = md_files[0].read_text()
        assert "type: Image Caption" in content
        assert "A quiet harbor at dusk." in content

        record = manifest.get(str(image_path))
        assert record.publish_status == "done"
        assert not image_path.exists()  # moved, not copied
    finally:
        manifest.close()


def test_publish_skips_images_not_yet_captioned(tmp_path: Path) -> None:
    image_path = tmp_path / "IMG_0002.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))
    output_dir = tmp_path / "output"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(image_path, "abcdef123456")
        manifest.update_stage(str(image_path), "raw", "done")
        # caption_status left at 'pending'

        run_publish(output_dir, manifest)

        assert list(output_dir.glob("*")) == []
        record = manifest.get(str(image_path))
        assert record.publish_status == "pending"
        assert image_path.exists()
    finally:
        manifest.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_publish.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.publish'`

- [ ] **Step 3: Write `src/image_captioner/publish.py`**

```python
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
            shutil.move(str(src_path), image_dest)
            doc = build_okf_document(
                title=record.title or "Untitled",
                caption=record.caption or "",
                tags=record.tags,
                image_relative_path=image_dest.name,
            )
            md_dest.write_text(doc, encoding="utf-8")
        except OSError as exc:
            manifest.update_stage(
                record.original_path, "publish", "failed", error_message=str(exc)
            )
            continue

        manifest.update_stage(
            record.original_path, "publish", "done", current_path=str(image_dest)
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_publish.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Wire the `publish` subcommand into the CLI**

Modify `src/image_captioner/cli.py`, adding:

```python
from image_captioner.publish import run_publish


@main.command()
@click.pass_obj
def publish(config: PipelineConfig) -> None:
    """Rename images and write OKF markdown into the output bundle."""
    manifest = Manifest(config.manifest_path)
    try:
        run_publish(config.output_dir, manifest)
    finally:
        manifest.close()
```

- [ ] **Step 6: Run the full test suite to verify nothing broke**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/image_captioner/publish.py src/image_captioner/cli.py tests/test_publish.py
git commit -m "feat: publish stage writing flat OKF bundle and CLI subcommand"
```

---

### Task 13: `run` convenience command and end-to-end test

**Files:**
- Modify: `src/image_captioner/cli.py`
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: all previously defined CLI commands (`dedup`, `convert_raw_cmd`, `caption`, `publish`) via `ctx.invoke`.
- Produces: `run` command added to the CLI group, running all four stages in order.

- [ ] **Step 1: Write the failing test in `tests/test_end_to_end.py`**

```python
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from image_captioner.cli import main
from image_captioner.vlm_client import CaptionResult
from tests.helpers import make_solid_image


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(f'base_dir = "{tmp_path}"\n')
    return config_path


def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_solid_image(input_dir / "photo.jpg", (400, 300), (10, 20, 30))
    config_path = _write_config(tmp_path)

    fake_result = CaptionResult(
        title="Calm Blue Room",
        caption="A calm, dimly lit blue-toned room. Empty and quiet.",
        tags=["calm", "blue", "interior"],
    )

    runner = CliRunner()
    with patch("image_captioner.caption.request_caption", return_value=fake_result):
        result = runner.invoke(main, ["--config", str(config_path), "run"])
        assert result.exit_code == 0, result.output

    output_dir = tmp_path / "output"
    md_files = list(output_dir.glob("*.md"))
    jpg_files = list(output_dir.glob("*.jpg"))
    assert len(md_files) == 1
    assert len(jpg_files) == 1
    assert "calm-blue-room" in md_files[0].stem

    content = md_files[0].read_text()
    assert "type: Image Caption" in content
    assert "A calm, dimly lit blue-toned room" in content
    assert f"![Calm Blue Room]({jpg_files[0].name})" in content
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_end_to_end.py -v`
Expected: FAIL — `run` is not a registered command (click reports "No such command 'run'").

- [ ] **Step 3: Add the `run` command to `src/image_captioner/cli.py`**

Append to the file:

```python
@main.command()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Run all stages in order: dedup, convert-raw, caption, publish."""
    ctx.invoke(dedup)
    ctx.invoke(convert_raw_cmd)
    ctx.invoke(caption)
    ctx.invoke(publish)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 5: Run the entire test suite**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/image_captioner/cli.py tests/test_end_to_end.py
git commit -m "feat: run convenience command chaining all stages, end-to-end test"
```

---

## Post-plan follow-ups (not part of this plan)

- If Task 2's smoke test shows JSON-constrained output *does* work reliably via llama-server grammars, revisit `vlm_client.request_caption` to pass a `json_schema`/`grammar` parameter for stronger guarantees — currently it deliberately relies only on lenient text extraction (`parse_caption_json`), matching the smoke test's validated baseline.
- Sub-project 2 (model evaluation harness) will run multiple `vlm_endpoint`/`vlm_prompt` configs over the same input set and needs the manifest schema extended (e.g. a `model_name` column) — out of scope here.
- Sub-project 3 (categorization/organization pass) will enrich `tags` and add cross-links across the flat OKF bundle produced by `publish` — out of scope here.
