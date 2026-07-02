# Model Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an `evaluate` CLI subcommand that captions a folder of images with every configured candidate VLM (auto-managing `llama-server` per candidate), scores every (image, model) caption pair with an OpenRouter LLM judge on a fixed rubric, and writes a markdown comparison report.

**Architecture:** A new `image_captioner.evaluation` package with five focused modules — config loading, `llama-server` process lifecycle, caption-phase orchestration, judge-phase orchestration, and report rendering — wired together by a new `evaluate` subcommand on the existing `image-captioner` CLI. The harness reuses the core pipeline's existing `vlm_client.request_caption`, `resize.resize_for_vlm`, and `config.DEFAULT_PROMPT` so captions are generated exactly as production would generate them. Results persist incrementally to a JSON file between the caption and judge phases, which also serves as a resume point via `--from-captions`.

**Tech Stack:** Python 3.11, `click` (CLI), `requests` (HTTP, both to `llama-server` and OpenRouter), `subprocess` (stdlib, `llama-server` lifecycle), `tomllib` (stdlib, config), `json` (stdlib, results persistence), `pytest` + `unittest.mock` (tests).

## Global Constraints

- Judge API: OpenRouter (`https://openrouter.ai/api/v1/chat/completions`), OpenAI-compatible chat-completions shape. API key read from the `OPENROUTER_API_KEY` environment variable at call time — never stored in `eval.toml` or committed anywhere.
- Captioning must go through the exact same code as production: `image_captioner.vlm_client.request_caption`, `image_captioner.resize.resize_for_vlm`, and `image_captioner.config.DEFAULT_PROMPT` as the default prompt.
- The harness owns the full `llama-server` lifecycle per candidate: launch via `scripts/run_llama_server.sh`, poll `/health` until ready or a timeout, run all images, then stop the server before moving to the next candidate. Only one candidate's server runs at a time.
- Scoring rubric: four 1–10 integer scores per (image, model) pair — `accuracy`, `descriptiveness`, `evocativeness`, `mood_fit` — plus a short `reasoning` string. Each pair is judged independently; no comparative/pairwise judging between models.
- `eval-results.json` is written incrementally (after every image in the caption phase, after every image in the judge phase) — never only at the end — so a crash loses at most one image's work and `--from-captions` can resume from it.
- No SQLite manifest. This is an occasional evaluation tool over a small image set, not the resumable production pipeline.
- Retries: up to `max_retries` (config, default 2) with exponential backoff (`2 ** attempt` seconds between attempts), same pattern as `caption.py`'s existing retry loop. Applies independently to both the caption phase and the judge phase.
- Error handling never aborts the whole run: a candidate whose server never becomes healthy is skipped (all its images recorded `failed`); a caption or judge failure after retries is recorded per-image and the run continues.
- CLI: `image-captioner evaluate --config eval.toml [--from-captions eval-results.json]`. Default `--config` value is `eval.toml` in the current directory.
- Out of scope for this plan: automatically sourcing new candidate models/mmproj files, comparative/pairwise judging, tag enrichment of eval results (sub-project 3's concern), any GUI.

---

## File Structure

```
src/image_captioner/
  evaluation/
    __init__.py
    config.py          # EvalConfig + Candidate, loaded from TOML
    server_manager.py  # start/stop llama-server subprocess, /health polling
    runner.py          # caption-phase orchestration; eval-results.json read/write
    judge.py            # OpenRouter judge client + rubric + judge-phase orchestration
    report.py            # markdown report generation
  cli.py                 # add `evaluate` subcommand; make pipeline config loading lazy
eval.toml.example
tests/
  test_eval_config.py
  test_server_manager.py
  test_eval_runner.py
  test_judge.py
  test_report.py
  test_cli.py            # extended: evaluate command + lazy-config regression tests
```

---

### Task 1: `evaluation/config.py` — EvalConfig and Candidate

**Files:**
- Create: `src/image_captioner/evaluation/__init__.py`
- Create: `src/image_captioner/evaluation/config.py`
- Test: `tests/test_eval_config.py`

**Interfaces:**
- Produces: `Candidate` dataclass (`name: str`, `model_path: Path`, `mmproj_path: Path | None = None`). `EvalConfig` dataclass with fields `candidates: list[Candidate]`, `image_dir: Path`, `judge_model: str`, `output_report: Path = Path("eval-report.md")`, `results_path: Path = Path("eval-results.json")`, `port: int = 8090`, `vlm_prompt: str = DEFAULT_PROMPT`, `resize_max_dim: int = 1568`, `resize_jpeg_quality: int = 92`, `max_retries: int = 2`, `server_startup_timeout: float = 120.0`. Classmethod `EvalConfig.from_toml(path: Path) -> EvalConfig`.

- [ ] **Step 1: Write `src/image_captioner/evaluation/__init__.py`** (empty file marking the package)

```python
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_eval_config.py
from pathlib import Path

from image_captioner.config import DEFAULT_PROMPT
from image_captioner.evaluation.config import Candidate, EvalConfig


def test_from_toml_parses_candidates_and_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
image_dir = "/tmp/images"
judge_model = "anthropic/claude-opus-4-8"

[[candidates]]
name = "qwen3-vl-8b"
model_path = "/models/qwen3-vl-8b.gguf"
mmproj_path = "/models/qwen3-vl-8b-mmproj.gguf"

[[candidates]]
name = "text-only-candidate"
model_path = "/models/text-only.gguf"
"""
    )

    config = EvalConfig.from_toml(config_path)

    assert config.image_dir == Path("/tmp/images")
    assert config.judge_model == "anthropic/claude-opus-4-8"
    assert config.candidates == [
        Candidate(
            name="qwen3-vl-8b",
            model_path=Path("/models/qwen3-vl-8b.gguf"),
            mmproj_path=Path("/models/qwen3-vl-8b-mmproj.gguf"),
        ),
        Candidate(
            name="text-only-candidate",
            model_path=Path("/models/text-only.gguf"),
            mmproj_path=None,
        ),
    ]


def test_from_toml_applies_defaults_for_optional_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
image_dir = "/tmp/images"
judge_model = "anthropic/claude-opus-4-8"

[[candidates]]
name = "solo"
model_path = "/models/solo.gguf"
"""
    )

    config = EvalConfig.from_toml(config_path)

    assert config.output_report == Path("eval-report.md")
    assert config.results_path == Path("eval-results.json")
    assert config.port == 8090
    assert config.vlm_prompt == DEFAULT_PROMPT
    assert config.resize_max_dim == 1568
    assert config.resize_jpeg_quality == 92
    assert config.max_retries == 2
    assert config.server_startup_timeout == 120.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.evaluation'`

- [ ] **Step 4: Write `src/image_captioner/evaluation/config.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/image_captioner/evaluation/__init__.py src/image_captioner/evaluation/config.py tests/test_eval_config.py
git commit -m "feat: add EvalConfig for the model evaluation harness"
```

---

### Task 2: `evaluation/server_manager.py` — llama-server lifecycle

**Files:**
- Create: `src/image_captioner/evaluation/server_manager.py`
- Test: `tests/test_server_manager.py`

**Interfaces:**
- Consumes: `Candidate` from `image_captioner.evaluation.config` (Task 1).
- Produces: `ServerStartupError(Exception)`. `start_server(candidate: Candidate, port: int) -> subprocess.Popen`. `wait_for_health(port: int, timeout: float, poll_interval: float = 1.0) -> None` (raises `ServerStartupError` on timeout). `stop_server(process: subprocess.Popen, timeout: float = 10.0) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_manager.py
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from image_captioner.evaluation.config import Candidate
from image_captioner.evaluation.server_manager import (
    ServerStartupError,
    start_server,
    stop_server,
    wait_for_health,
)


def test_start_server_includes_mmproj_and_port_env() -> None:
    candidate = Candidate(
        name="qwen3-vl-8b",
        model_path=Path("/models/qwen3-vl-8b.gguf"),
        mmproj_path=Path("/models/qwen3-vl-8b-mmproj.gguf"),
    )

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        start_server(candidate, port=8091)

    args, kwargs = mock_popen.call_args
    command = args[0]
    assert command[-2:] == ["/models/qwen3-vl-8b.gguf", "/models/qwen3-vl-8b-mmproj.gguf"]
    assert kwargs["env"]["LLAMA_SERVER_PORT"] == "8091"


def test_start_server_omits_mmproj_when_none() -> None:
    candidate = Candidate(name="text-only", model_path=Path("/models/text-only.gguf"))

    with patch("image_captioner.evaluation.server_manager.subprocess.Popen") as mock_popen:
        start_server(candidate, port=8092)

    args, _ = mock_popen.call_args
    command = args[0]
    assert command[-1] == "/models/text-only.gguf"


def test_wait_for_health_returns_when_server_responds_ok() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 200

    with patch(
        "image_captioner.evaluation.server_manager.requests.get", return_value=fake_response
    ):
        wait_for_health(port=8091, timeout=5.0, poll_interval=0.01)


def test_wait_for_health_raises_after_timeout() -> None:
    with patch(
        "image_captioner.evaluation.server_manager.requests.get",
        side_effect=requests.ConnectionError("not up yet"),
    ):
        with pytest.raises(ServerStartupError):
            wait_for_health(port=8091, timeout=0.05, poll_interval=0.01)


def test_stop_server_terminates_and_waits() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.wait.return_value = 0

    stop_server(process)

    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    process.kill.assert_not_called()


def test_stop_server_kills_when_terminate_times_out() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=10.0), 0]

    stop_server(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.evaluation.server_manager'`

- [ ] **Step 3: Write `src/image_captioner/evaluation/server_manager.py`**

```python
"""Manages the lifecycle of a local llama-server subprocess per candidate model."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import requests

from image_captioner.evaluation.config import Candidate

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_llama_server.sh"


class ServerStartupError(Exception):
    """Raised when llama-server does not become healthy within the timeout."""


def start_server(candidate: Candidate, port: int) -> subprocess.Popen:
    args = ["bash", str(SCRIPT_PATH), str(candidate.model_path)]
    if candidate.mmproj_path is not None:
        args.append(str(candidate.mmproj_path))
    env = dict(os.environ)
    env["LLAMA_SERVER_PORT"] = str(port)
    return subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_health(port: int, timeout: float, poll_interval: float = 1.0) -> None:
    endpoint = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(endpoint, timeout=2.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    raise ServerStartupError(
        f"llama-server did not become healthy within {timeout}s on port {port}"
    )


def stop_server(process: subprocess.Popen, timeout: float = 10.0) -> None:
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server_manager.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/evaluation/server_manager.py tests/test_server_manager.py
git commit -m "feat: add llama-server lifecycle management for evaluation candidates"
```

---

### Task 3: `evaluation/runner.py` — caption-phase orchestration

**Files:**
- Create: `src/image_captioner/evaluation/runner.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `EvalConfig`, `Candidate` (Task 1); `ServerStartupError`, `start_server`, `wait_for_health`, `stop_server` (Task 2); existing `IMAGE_EXTENSIONS` from `image_captioner.formats`; existing `resize_for_vlm` from `image_captioner.resize`; existing `request_caption`, `VLMResponseError` from `image_captioner.vlm_client`.
- Produces: `discover_images(image_dir: Path) -> list[Path]`. `write_results(path: Path, results: dict) -> None`. `load_results(path: Path) -> dict`. `run_captioning(config: EvalConfig) -> dict`, where the returned/written results dict has shape `{"candidates": [str, ...], "images": [str, ...], "results": {candidate_name: {image_key: record}}}` and each `record` is `{"status": "done"|"failed", "title": str|None, "caption": str|None, "tags": list[str]|None, "elapsed_seconds": float|None, "error": str|None, "scores": dict|None}`. `image_key` is the image path relative to `config.image_dir`, as a string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_runner.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from image_captioner.evaluation.config import Candidate, EvalConfig
from image_captioner.evaluation.runner import (
    discover_images,
    load_results,
    run_captioning,
    write_results,
)
from image_captioner.evaluation.server_manager import ServerStartupError
from image_captioner.vlm_client import CaptionResult
from tests.helpers import make_solid_image


def test_discover_images_finds_and_sorts_supported_extensions(tmp_path: Path) -> None:
    make_solid_image(tmp_path / "b.jpg", (100, 100), (10, 20, 30))
    make_solid_image(tmp_path / "a.png", (100, 100), (10, 20, 30))
    (tmp_path / "ignore.txt").write_text("not an image")

    images = discover_images(tmp_path)

    assert [p.name for p in images] == ["a.png", "b.jpg"]


def test_write_results_and_load_results_round_trip(tmp_path: Path) -> None:
    results_path = tmp_path / "eval-results.json"
    results = {"candidates": ["a"], "images": ["x.jpg"], "results": {"a": {}}}

    write_results(results_path, results)
    loaded = load_results(results_path)

    assert loaded == results


def _config(tmp_path: Path, image_dir: Path) -> EvalConfig:
    return EvalConfig(
        candidates=[Candidate(name="cand-a", model_path=tmp_path / "a.gguf")],
        image_dir=image_dir,
        judge_model="anthropic/claude-opus-4-8",
        output_report=tmp_path / "eval-report.md",
        results_path=tmp_path / "eval-results.json",
        port=8099,
        max_retries=0,
    )


def test_run_captioning_writes_results_for_each_candidate_and_image(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    make_solid_image(image_dir / "photo.jpg", (200, 150), (10, 20, 30))
    config = _config(tmp_path, image_dir)

    fake_result = CaptionResult(title="T", caption="C", tags=["x"])
    fake_process = MagicMock()

    with patch(
        "image_captioner.evaluation.runner.start_server", return_value=fake_process
    ) as mock_start, patch(
        "image_captioner.evaluation.runner.wait_for_health"
    ) as mock_wait, patch(
        "image_captioner.evaluation.runner.stop_server"
    ) as mock_stop, patch(
        "image_captioner.evaluation.runner.request_caption", return_value=fake_result
    ):
        results = run_captioning(config)

    mock_start.assert_called_once_with(config.candidates[0], config.port)
    mock_wait.assert_called_once_with(config.port, config.server_startup_timeout)
    mock_stop.assert_called_once_with(fake_process)

    record = results["results"]["cand-a"]["photo.jpg"]
    assert record["status"] == "done"
    assert record["title"] == "T"
    assert record["caption"] == "C"
    assert record["tags"] == ["x"]
    assert record["elapsed_seconds"] is not None

    on_disk = load_results(config.results_path)
    assert on_disk == results


def test_run_captioning_skips_candidate_when_server_never_becomes_healthy(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    make_solid_image(image_dir / "photo.jpg", (200, 150), (10, 20, 30))
    config = _config(tmp_path, image_dir)

    fake_process = MagicMock()

    with patch(
        "image_captioner.evaluation.runner.start_server", return_value=fake_process
    ), patch(
        "image_captioner.evaluation.runner.wait_for_health",
        side_effect=ServerStartupError("never came up"),
    ) as mock_wait, patch(
        "image_captioner.evaluation.runner.stop_server"
    ) as mock_stop, patch(
        "image_captioner.evaluation.runner.request_caption"
    ) as mock_request:
        results = run_captioning(config)

    mock_wait.assert_called_once()
    mock_stop.assert_called_once_with(fake_process)
    mock_request.assert_not_called()

    record = results["results"]["cand-a"]["photo.jpg"]
    assert record["status"] == "failed"
    assert "never came up" in record["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.evaluation.runner'`

- [ ] **Step 3: Write `src/image_captioner/evaluation/runner.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/evaluation/runner.py tests/test_eval_runner.py
git commit -m "feat: add caption-phase orchestration for the evaluation harness"
```

---

### Task 4: `evaluation/judge.py` — OpenRouter judge and judge-phase orchestration

**Files:**
- Create: `src/image_captioner/evaluation/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `EvalConfig`, `Candidate` (Task 1); `write_results` (Task 3, `image_captioner.evaluation.runner`).
- Produces: `JudgeResponseError(Exception)`. `JudgeScores` dataclass (`accuracy: int`, `descriptiveness: int`, `evocativeness: int`, `mood_fit: int`, `reasoning: str`). `parse_judge_json(content: str) -> JudgeScores`. `request_judge_score(judge_model: str, image_path: Path, title: str, caption: str, timeout: float = 120.0) -> JudgeScores`. `run_judging(config: EvalConfig, results: dict) -> dict` — mutates and returns `results`, filling in each `done`-status record's `scores` key (or leaving it `None` and setting a `judge_error` key on exhausted retries).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_captioner.evaluation.config import Candidate, EvalConfig
from image_captioner.evaluation.judge import (
    JudgeResponseError,
    JudgeScores,
    parse_judge_json,
    request_judge_score,
    run_judging,
)
from image_captioner.evaluation.runner import load_results


def test_parse_judge_json_clean() -> None:
    content = (
        '{"accuracy": 8, "descriptiveness": 7, "evocativeness": 9, '
        '"mood_fit": 6, "reasoning": "Vivid and accurate."}'
    )
    result = parse_judge_json(content)
    assert result == JudgeScores(
        accuracy=8,
        descriptiveness=7,
        evocativeness=9,
        mood_fit=6,
        reasoning="Vivid and accurate.",
    )


def test_parse_judge_json_missing_key_raises() -> None:
    with pytest.raises(JudgeResponseError):
        parse_judge_json('{"accuracy": 8, "descriptiveness": 7}')


def test_parse_judge_json_malformed_raises() -> None:
    with pytest.raises(JudgeResponseError):
        parse_judge_json("not json at all")


def test_request_judge_score_posts_expected_payload_and_parses_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"accuracy": 9, "descriptiveness": 8, "evocativeness": 7, '
                        '"mood_fit": 6, "reasoning": "Great detail."}'
                    )
                }
            }
        ]
    }

    with patch(
        "image_captioner.evaluation.judge.requests.post", return_value=fake_response
    ) as post:
        result = request_judge_score(
            "anthropic/claude-opus-4-8", image_path, "A Title", "A caption body."
        )

    assert result.accuracy == 9
    assert result.reasoning == "Great detail."
    called_url = post.call_args.args[0]
    assert called_url == "https://openrouter.ai/api/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "anthropic/claude-opus-4-8"
    content_parts = payload["messages"][0]["content"]
    assert "A Title" in content_parts[0]["text"]
    assert content_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_request_judge_score_raises_when_api_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    with pytest.raises(JudgeResponseError):
        request_judge_score("anthropic/claude-opus-4-8", image_path, "T", "C")


def _config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        candidates=[Candidate(name="cand-a", model_path=tmp_path / "a.gguf")],
        image_dir=tmp_path,
        judge_model="anthropic/claude-opus-4-8",
        output_report=tmp_path / "eval-report.md",
        results_path=tmp_path / "eval-results.json",
        max_retries=0,
    )


def test_run_judging_scores_done_records_and_skips_others(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"fake-jpeg-bytes")
    config = _config(tmp_path)
    results = {
        "candidates": ["cand-a"],
        "images": ["photo.jpg", "failed.jpg", "already-scored.jpg"],
        "results": {
            "cand-a": {
                "photo.jpg": {
                    "status": "done",
                    "title": "T",
                    "caption": "C",
                    "tags": [],
                    "elapsed_seconds": 1.0,
                    "error": None,
                    "scores": None,
                },
                "failed.jpg": {
                    "status": "failed",
                    "title": None,
                    "caption": None,
                    "tags": None,
                    "elapsed_seconds": None,
                    "error": "boom",
                    "scores": None,
                },
                "already-scored.jpg": {
                    "status": "done",
                    "title": "T2",
                    "caption": "C2",
                    "tags": [],
                    "elapsed_seconds": 1.0,
                    "error": None,
                    "scores": {
                        "accuracy": 5,
                        "descriptiveness": 5,
                        "evocativeness": 5,
                        "mood_fit": 5,
                        "reasoning": "already done",
                    },
                },
            }
        },
    }

    fake_scores = JudgeScores(
        accuracy=8, descriptiveness=7, evocativeness=9, mood_fit=6, reasoning="Good."
    )
    with patch(
        "image_captioner.evaluation.judge.request_judge_score", return_value=fake_scores
    ) as mock_request:
        run_judging(config, results)

    mock_request.assert_called_once()
    assert results["results"]["cand-a"]["photo.jpg"]["scores"]["accuracy"] == 8
    assert results["results"]["cand-a"]["failed.jpg"]["scores"] is None
    assert (
        results["results"]["cand-a"]["already-scored.jpg"]["scores"]["reasoning"]
        == "already done"
    )

    on_disk = load_results(config.results_path)
    assert on_disk == results


@patch("image_captioner.evaluation.judge.time.sleep")
def test_run_judging_marks_unscored_after_retries_exhausted(mock_sleep, tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"fake-jpeg-bytes")
    config = _config(tmp_path)
    results = {
        "candidates": ["cand-a"],
        "images": ["photo.jpg"],
        "results": {
            "cand-a": {
                "photo.jpg": {
                    "status": "done",
                    "title": "T",
                    "caption": "C",
                    "tags": [],
                    "elapsed_seconds": 1.0,
                    "error": None,
                    "scores": None,
                },
            }
        },
    }

    with patch(
        "image_captioner.evaluation.judge.request_judge_score",
        side_effect=JudgeResponseError("bad json"),
    ):
        run_judging(config, results)

    record = results["results"]["cand-a"]["photo.jpg"]
    assert record["scores"] is None
    assert "bad json" in record["judge_error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.evaluation.judge'`

- [ ] **Step 3: Write `src/image_captioner/evaluation/judge.py`**

```python
"""OpenRouter-backed LLM judge for scoring generated captions."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from image_captioner.evaluation.config import EvalConfig
from image_captioner.evaluation.runner import write_results

logger = logging.getLogger(__name__)

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

RUBRIC_PROMPT_TEMPLATE = (
    "You are scoring an AI-generated image caption for quality. Look at the "
    "image and the caption below, then score the caption on four axes, each "
    "from 1 (poor) to 10 (excellent):\n"
    "- accuracy: does the caption match what is actually in the image\n"
    "- descriptiveness: level of detail\n"
    "- evocativeness: quality of the descriptive/interpretive language\n"
    "- mood_fit: how well the caption captures the atmosphere/mood of the "
    "image, in a way that would help match this image to a piece of music "
    "by its vibe\n\n"
    "Caption title: {title}\n"
    "Caption body: {caption}\n\n"
    "Respond ONLY with JSON in this exact shape: "
    '{{"accuracy": <1-10>, "descriptiveness": <1-10>, "evocativeness": <1-10>, '
    '"mood_fit": <1-10>, "reasoning": "<one or two sentences>"}}'
)


class JudgeResponseError(Exception):
    """Raised when the judge response can't be parsed into the expected shape."""


@dataclass
class JudgeScores:
    accuracy: int
    descriptiveness: int
    evocativeness: int
    mood_fit: int
    reasoning: str


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def parse_judge_json(content: str) -> JudgeScores:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeResponseError(f"no JSON object found in content: {content!r}")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"invalid JSON in content: {content!r}") from exc
    for key in ("accuracy", "descriptiveness", "evocativeness", "mood_fit", "reasoning"):
        if key not in data:
            raise JudgeResponseError(f"missing key {key!r} in judge JSON: {data!r}")
    return JudgeScores(
        accuracy=int(data["accuracy"]),
        descriptiveness=int(data["descriptiveness"]),
        evocativeness=int(data["evocativeness"]),
        mood_fit=int(data["mood_fit"]),
        reasoning=str(data["reasoning"]),
    )


def request_judge_score(
    judge_model: str, image_path: Path, title: str, caption: str, timeout: float = 120.0
) -> JudgeScores:
    try:
        api_key = os.environ["OPENROUTER_API_KEY"]
    except KeyError as exc:
        raise JudgeResponseError("OPENROUTER_API_KEY environment variable is not set") from exc

    b64 = _encode_image(image_path)
    prompt = RUBRIC_PROMPT_TEMPLATE.format(title=title, caption=caption)
    payload = {
        "model": judge_model,
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
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(OPENROUTER_ENDPOINT, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise JudgeResponseError(f"unexpected response shape: {body}") from exc
    return parse_judge_json(content)


def run_judging(config: EvalConfig, results: dict) -> dict:
    for candidate in config.candidates:
        candidate_results = results["results"].get(candidate.name, {})
        for image_key, record in candidate_results.items():
            if record["status"] != "done" or record.get("scores") is not None:
                continue

            image_path = config.image_dir / image_key
            last_error: Exception | None = None
            for attempt in range(config.max_retries + 1):
                try:
                    scores = request_judge_score(
                        config.judge_model, image_path, record["title"], record["caption"]
                    )
                    record["scores"] = {
                        "accuracy": scores.accuracy,
                        "descriptiveness": scores.descriptiveness,
                        "evocativeness": scores.evocativeness,
                        "mood_fit": scores.mood_fit,
                        "reasoning": scores.reasoning,
                    }
                    record["judge_error"] = None
                    last_error = None
                    break
                except (JudgeResponseError, OSError, requests.RequestException) as exc:
                    last_error = exc
                    logger.warning(
                        "judge attempt %d/%d failed for %s/%s: %s",
                        attempt + 1,
                        config.max_retries + 1,
                        candidate.name,
                        image_key,
                        exc,
                    )
                    if attempt < config.max_retries:
                        time.sleep(2**attempt)

            if last_error is not None:
                record["scores"] = None
                record["judge_error"] = str(last_error)

            write_results(config.results_path, results)

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_judge.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/evaluation/judge.py tests/test_judge.py
git commit -m "feat: add OpenRouter judge and judge-phase orchestration"
```

---

### Task 5: `evaluation/report.py` — markdown report generation

**Files:**
- Create: `src/image_captioner/evaluation/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: results dict shape produced by Tasks 3/4 (`{"candidates": [...], "images": [...], "results": {name: {image_key: record}}}`; `record` may additionally carry a `judge_error` key).
- Produces: `compute_model_summary(candidate_name: str, candidate_results: dict) -> dict` returning `{"name", "n_images", "n_captioned", "n_scored", "avg_accuracy", "avg_descriptiveness", "avg_evocativeness", "avg_quality", "avg_mood_fit", "avg_seconds"}` (averages are `float | None`; `None` when there is nothing to average). `render_report(results: dict) -> str`. `write_report(results: dict, output_path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from pathlib import Path

from image_captioner.evaluation.report import (
    compute_model_summary,
    render_report,
    write_report,
)


def _sample_results() -> dict:
    return {
        "candidates": ["cand-a"],
        "images": ["photo.jpg", "unscored.jpg"],
        "results": {
            "cand-a": {
                "photo.jpg": {
                    "status": "done",
                    "title": "Whispers of Dawn",
                    "caption": "A quiet lake at sunrise.",
                    "tags": ["lake", "dawn"],
                    "elapsed_seconds": 2.5,
                    "error": None,
                    "scores": {
                        "accuracy": 8,
                        "descriptiveness": 7,
                        "evocativeness": 9,
                        "mood_fit": 6,
                        "reasoning": "Evocative and accurate.",
                    },
                },
                "unscored.jpg": {
                    "status": "done",
                    "title": "Other",
                    "caption": "Something else.",
                    "tags": [],
                    "elapsed_seconds": 1.0,
                    "error": None,
                    "scores": None,
                    "judge_error": "OPENROUTER_API_KEY environment variable is not set",
                },
            }
        },
    }


def test_compute_model_summary_averages_scores_and_counts() -> None:
    results = _sample_results()
    summary = compute_model_summary("cand-a", results["results"]["cand-a"])

    assert summary["n_images"] == 2
    assert summary["n_captioned"] == 2
    assert summary["n_scored"] == 1
    assert summary["avg_accuracy"] == 8.0
    assert summary["avg_quality"] == (8 + 7 + 9) / 3
    assert summary["avg_mood_fit"] == 6.0
    assert summary["avg_seconds"] == (2.5 + 1.0) / 2


def test_compute_model_summary_handles_no_scored_images() -> None:
    candidate_results = {
        "photo.jpg": {
            "status": "failed",
            "title": None,
            "caption": None,
            "tags": None,
            "elapsed_seconds": None,
            "error": "boom",
            "scores": None,
        },
    }
    summary = compute_model_summary("cand-a", candidate_results)

    assert summary["n_captioned"] == 0
    assert summary["n_scored"] == 0
    assert summary["avg_accuracy"] is None
    assert summary["avg_seconds"] is None


def test_render_report_includes_summary_table_and_per_image_detail() -> None:
    report = render_report(_sample_results())

    assert "| cand-a |" in report
    assert "Whispers of Dawn" in report
    assert "accuracy=8" in report
    assert "unscored" in report
    assert "OPENROUTER_API_KEY environment variable is not set" in report


def test_write_report_writes_file(tmp_path: Path) -> None:
    output_path = tmp_path / "eval-report.md"
    write_report(_sample_results(), output_path)

    assert output_path.exists()
    assert "cand-a" in output_path.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_captioner.evaluation.report'`

- [ ] **Step 3: Write `src/image_captioner/evaluation/report.py`**

```python
"""Markdown report generation from evaluation results."""
from __future__ import annotations

from pathlib import Path


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_model_summary(candidate_name: str, candidate_results: dict) -> dict:
    accuracy: list[float] = []
    descriptiveness: list[float] = []
    evocativeness: list[float] = []
    mood_fit: list[float] = []
    quality: list[float] = []
    elapsed: list[float] = []
    n_captioned = 0
    n_scored = 0

    for record in candidate_results.values():
        if record["status"] == "done":
            n_captioned += 1
            if record["elapsed_seconds"] is not None:
                elapsed.append(record["elapsed_seconds"])
        scores = record.get("scores")
        if scores is not None:
            n_scored += 1
            accuracy.append(scores["accuracy"])
            descriptiveness.append(scores["descriptiveness"])
            evocativeness.append(scores["evocativeness"])
            mood_fit.append(scores["mood_fit"])
            quality.append(
                (scores["accuracy"] + scores["descriptiveness"] + scores["evocativeness"]) / 3
            )

    return {
        "name": candidate_name,
        "n_images": len(candidate_results),
        "n_captioned": n_captioned,
        "n_scored": n_scored,
        "avg_accuracy": _average(accuracy),
        "avg_descriptiveness": _average(descriptiveness),
        "avg_evocativeness": _average(evocativeness),
        "avg_quality": _average(quality),
        "avg_mood_fit": _average(mood_fit),
        "avg_seconds": _average(elapsed),
    }


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def render_report(results: dict) -> str:
    lines = ["# Model Evaluation Report", ""]
    lines.append(
        "| Model | Images | Captioned | Scored | Accuracy | Descriptiveness | "
        "Evocativeness | Quality | Mood Fit | Avg sec/image |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    summaries = [
        compute_model_summary(name, results["results"][name]) for name in results["candidates"]
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['n_images']} | {s['n_captioned']} | {s['n_scored']} | "
            f"{_fmt(s['avg_accuracy'])} | {_fmt(s['avg_descriptiveness'])} | "
            f"{_fmt(s['avg_evocativeness'])} | {_fmt(s['avg_quality'])} | "
            f"{_fmt(s['avg_mood_fit'])} | {_fmt(s['avg_seconds'])} |"
        )

    lines.append("")
    lines.append("## Per-image detail")
    for image in results["images"]:
        lines.append("")
        lines.append(f"### {image}")
        for name in results["candidates"]:
            record = results["results"][name].get(image)
            if record is None:
                continue
            lines.append("")
            lines.append(f"**{name}**")
            if record["status"] != "done":
                lines.append(f"- captioning failed: {record['error']}")
                continue
            lines.append(f"- title: {record['title']}")
            lines.append(f"- caption: {record['caption']}")
            scores = record.get("scores")
            if scores is None:
                judge_error = record.get("judge_error")
                suffix = f": {judge_error}" if judge_error else ""
                lines.append(f"- unscored{suffix}")
            else:
                lines.append(
                    f"- scores: accuracy={scores['accuracy']}, "
                    f"descriptiveness={scores['descriptiveness']}, "
                    f"evocativeness={scores['evocativeness']}, "
                    f"mood_fit={scores['mood_fit']}"
                )
                lines.append(f"- reasoning: {scores['reasoning']}")

    return "\n".join(lines) + "\n"


def write_report(results: dict, output_path: Path) -> None:
    output_path.write_text(render_report(results))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_captioner/evaluation/report.py tests/test_report.py
git commit -m "feat: add markdown report generation for the evaluation harness"
```

---

### Task 6: CLI wiring — `evaluate` subcommand and lazy pipeline config loading

**Context for the implementer:** The existing `main` click group eagerly loads `PipelineConfig` from `--config` (default `pipeline.toml`) in the group callback, and `click.Path(exists=True, ...)` on that option means **any** invocation of the CLI — including a future `evaluate` subcommand that doesn't touch `pipeline.toml` at all — fails unless a `pipeline.toml` happens to exist in the current directory. This task fixes that by making the group callback store the raw config *path* instead of eagerly loading it, and having each pipeline subcommand (`dedup`, `convert-raw`, `caption`, `publish`, `run`) load `PipelineConfig` itself. The `evaluate` subcommand gets its own independent `--config` option (default `eval.toml`) and never touches the pipeline's config path.

**Files:**
- Modify: `src/image_captioner/cli.py`
- Modify: `tests/test_cli.py`
- Create: `eval.toml.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: `EvalConfig.from_toml` (Task 1); `run_captioning`, `load_results` (Task 3); `run_judging` (Task 4); `write_report` (Task 5); existing `PipelineConfig`, `run_caption`, `run_dedup`, `run_convert_raw`, `run_publish`, `Manifest`.
- Produces: `image-captioner evaluate --config eval.toml [--from-captions PATH]` CLI command.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (the existing `test_cli_help_runs` stays as-is; add these below it):

```python
def test_dedup_command_errors_cleanly_when_pipeline_config_missing(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_path = tmp_path / "does-not-exist.toml"
    result = runner.invoke(main, ["--config", str(missing_path), "dedup"])

    assert result.exit_code != 0
    assert "config file not found" in result.output


def test_evaluate_does_not_require_pipeline_toml(tmp_path: Path) -> None:
    """`evaluate` must work in a directory with no pipeline.toml at all."""
    eval_config_path = tmp_path / "eval.toml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    eval_config_path.write_text(
        f"""
image_dir = "{image_dir}"
judge_model = "anthropic/claude-opus-4-8"
output_report = "{tmp_path / 'eval-report.md'}"
results_path = "{tmp_path / 'eval-results.json'}"

[[candidates]]
name = "cand-a"
model_path = "/models/a.gguf"
"""
    )

    fake_results = {"candidates": ["cand-a"], "images": [], "results": {"cand-a": {}}}

    runner = CliRunner()
    with patch(
        "image_captioner.cli.run_captioning", return_value=fake_results
    ) as mock_run_captioning, patch(
        "image_captioner.cli.run_judging", side_effect=lambda config, results: results
    ):
        result = runner.invoke(main, ["evaluate", "--config", str(eval_config_path)])

    assert result.exit_code == 0, result.output
    mock_run_captioning.assert_called_once()
    assert (tmp_path / "eval-report.md").exists()


def test_evaluate_from_captions_skips_captioning_phase(tmp_path: Path) -> None:
    eval_config_path = tmp_path / "eval.toml"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    eval_config_path.write_text(
        f"""
image_dir = "{image_dir}"
judge_model = "anthropic/claude-opus-4-8"
output_report = "{tmp_path / 'eval-report.md'}"

[[candidates]]
name = "cand-a"
model_path = "/models/a.gguf"
"""
    )
    captions_path = tmp_path / "existing-results.json"
    captions_path.write_text(
        '{"candidates": ["cand-a"], "images": [], "results": {"cand-a": {}}}'
    )

    runner = CliRunner()
    with patch("image_captioner.cli.run_captioning") as mock_run_captioning, patch(
        "image_captioner.cli.run_judging", side_effect=lambda config, results: results
    ):
        result = runner.invoke(
            main,
            [
                "evaluate",
                "--config",
                str(eval_config_path),
                "--from-captions",
                str(captions_path),
            ],
        )

    assert result.exit_code == 0, result.output
    mock_run_captioning.assert_not_called()
```

Add the required imports to the top of `tests/test_cli.py` so the file reads:

```python
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from image_captioner.cli import main


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
```

(followed by the four new test functions above.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `test_dedup_command_errors_cleanly_when_pipeline_config_missing` fails because the current group raises a `click.exceptions.BadParameter`/usage error rather than "config file not found"; the two `evaluate` tests fail with `Error: No such command 'evaluate'`.

- [ ] **Step 3: Rewrite `src/image_captioner/cli.py`**

```python
"""Command-line entrypoint for the image-captioner pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from image_captioner.config import PipelineConfig
from image_captioner.evaluation.config import EvalConfig
from image_captioner.evaluation.judge import run_judging
from image_captioner.evaluation.report import write_report
from image_captioner.evaluation.runner import load_results, run_captioning


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("pipeline.toml"),
    show_default=True,
)
@click.pass_context
def main(ctx: click.Context, config_path: Path) -> None:
    """Turn a folder of images into captioned, renamed OKF markdown notes."""
    ctx.obj = config_path


from image_captioner.caption import run_caption
from image_captioner.dedup import run_dedup
from image_captioner.manifest import Manifest
from image_captioner.publish import run_publish
from image_captioner.raw_convert import run_convert_raw


def _load_pipeline_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise click.ClickException(f"config file not found: {config_path}")
    return PipelineConfig.from_toml(config_path)


def _print_stage_summary(manifest: Manifest, label: str, status_field: str) -> int:
    """Print a one-line done/failed/skipped summary for a stage and return the failed count."""
    counts = manifest.status_counts(status_field)
    n_done = counts.get("done", 0)
    n_failed = counts.get("failed", 0)
    n_skipped = counts.get("skipped", 0)
    click.echo(f"{label}: {n_done} done, {n_failed} failed, {n_skipped} skipped")
    return n_failed


def _do_dedup(config: PipelineConfig) -> int:
    manifest = Manifest(config.manifest_path)
    try:
        run_dedup(config.input_dir, config.duplicates_dir, manifest, config.phash_max_distance)
        return _print_stage_summary(manifest, "dedup", "dedup")
    finally:
        manifest.close()


def _do_convert_raw(config: PipelineConfig) -> int:
    manifest = Manifest(config.manifest_path)
    try:
        run_convert_raw(config.raw_originals_dir, manifest)
        return _print_stage_summary(manifest, "convert-raw", "raw")
    finally:
        manifest.close()


def _do_caption(config: PipelineConfig) -> int:
    manifest = Manifest(config.manifest_path)
    try:
        run_caption(config, manifest)
        return _print_stage_summary(manifest, "caption", "caption")
    finally:
        manifest.close()


def _do_publish(config: PipelineConfig) -> int:
    manifest = Manifest(config.manifest_path)
    try:
        run_publish(config.output_dir, manifest)
        return _print_stage_summary(manifest, "publish", "publish")
    finally:
        manifest.close()


@main.command()
@click.pass_obj
def dedup(config_path: Path) -> None:
    """Find and archive near-duplicate images."""
    config = _load_pipeline_config(config_path)
    if _do_dedup(config):
        sys.exit(1)


@main.command(name="convert-raw")
@click.pass_obj
def convert_raw_cmd(config_path: Path) -> None:
    """Convert RAW originals to JPEG and archive the originals."""
    config = _load_pipeline_config(config_path)
    if _do_convert_raw(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def caption(config_path: Path) -> None:
    """Caption each pending, raw-converted image via the local VLM."""
    config = _load_pipeline_config(config_path)
    if _do_caption(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def publish(config_path: Path) -> None:
    """Rename images and write OKF markdown into the output bundle."""
    config = _load_pipeline_config(config_path)
    if _do_publish(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def run(config_path: Path) -> None:
    """Run all stages in order: dedup, convert-raw, caption, publish."""
    config = _load_pipeline_config(config_path)
    n_failed = 0
    n_failed += _do_dedup(config)
    n_failed += _do_convert_raw(config)
    n_failed += _do_caption(config)
    n_failed += _do_publish(config)
    if n_failed:
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "eval_config_path",
    type=click.Path(path_type=Path),
    default=Path("eval.toml"),
    show_default=True,
)
@click.option(
    "--from-captions",
    "captions_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Skip captioning and re-run judging + report against an existing eval-results.json.",
)
def evaluate(eval_config_path: Path, captions_path: Path | None) -> None:
    """Caption a folder of images with each candidate VLM and score them with an LLM judge."""
    if not eval_config_path.exists():
        raise click.ClickException(f"config file not found: {eval_config_path}")
    config = EvalConfig.from_toml(eval_config_path)

    if captions_path is not None:
        results = load_results(captions_path)
    else:
        results = run_captioning(config)

    results = run_judging(config, results)
    write_report(results, config.output_report)
    click.echo(f"Evaluation report written to {config.output_report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py tests/test_end_to_end.py -v`
Expected: PASS — all `test_cli.py` tests pass, and `test_end_to_end.py`'s two tests (which pass `--config` explicitly to `run`) still pass unchanged.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — every test in the repo, old and new.

- [ ] **Step 6: Write `eval.toml.example`**

```toml
# Example configuration for the image-captioner model evaluation harness.
# Copy this file to eval.toml and adjust paths for your machine, then run:
#   uv run image-captioner evaluate --config eval.toml
#
# The judge calls OpenRouter; set OPENROUTER_API_KEY in your environment
# before running (never put it in this file).

# Folder of images to caption and score. Add or remove images anytime -
# the harness just processes whatever's here at run time.
image_dir = "/home/you/Pictures/test_pics"

# OpenRouter model id used as the scoring judge.
judge_model = "anthropic/claude-opus-4-8"

# Where the markdown comparison report is written.
output_report = "eval-report.md"

# Where captions + timings + judge scores are persisted, incrementally,
# as the run progresses. Reusable via `--from-captions` to re-run just
# the judge + report phases without re-captioning.
results_path = "eval-results.json"

# Local port the harness launches each candidate's llama-server on.
# Deliberately different from the pipeline's default 8080 so an eval run
# doesn't collide with a llama-server you already have running.
port = 8090

# Candidate VLMs to compare. Add as many as you like - the harness starts
# and stops llama-server once per candidate, in order.
[[candidates]]
name = "qwen3-vl-8b"
model_path = "/mnt/models/hf_ggufs/vlm/unsloth_Qwen3-VL-8B-Instruct-BF16.gguf"
mmproj_path = "/mnt/models/hf_ggufs/vlm/mmproj/qwen3vl-8b-mmproj-F16.gguf"

[[candidates]]
name = "joycaption-beta-one"
model_path = "/mnt/models/hf_ggufs/vlm/llama-joycaption-beta-one-hf-llava.f16.gguf"
mmproj_path = "/mnt/models/hf_ggufs/vlm/mmproj/joycaption-mmproj-f16.gguf"
```

- [ ] **Step 7: Add an "Evaluating candidate models" section to `README.md`**

Append this section to the end of `README.md`:

```markdown
## Evaluating candidate models

Before settling on a default VLM, compare candidates against a shared set of
images. Copy `eval.toml.example` to `eval.toml`, list the candidate models
you want to compare, and set `OPENROUTER_API_KEY` in your environment (the
harness uses OpenRouter as an independent LLM judge — never a candidate
model judging itself):

\`\`\`bash
export OPENROUTER_API_KEY=sk-or-...
uv run image-captioner evaluate --config eval.toml
\`\`\`

The harness starts `llama-server` for each candidate in turn (reusing
`scripts/run_llama_server.sh`), captions every image in `image_dir`, then
scores every caption with the judge model on four axes: accuracy,
descriptiveness, evocativeness, and mood fit. Results are written
incrementally to `eval-results.json` as the run progresses, and a markdown
comparison report is written to `output_report` (`eval-report.md` by
default) when it finishes.

If you only want to re-run the judge and report — for example, after
tweaking the rubric — skip the expensive local captioning phase:

\`\`\`bash
uv run image-captioner evaluate --config eval.toml --from-captions eval-results.json
\`\`\`
```

- [ ] **Step 8: Commit**

```bash
git add src/image_captioner/cli.py tests/test_cli.py eval.toml.example README.md
git commit -m "feat: add evaluate CLI command and make pipeline config loading lazy"
```

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 6 CLI + Task 1 config), all five components (Tasks 1-5), scoring rubric (Task 4's `RUBRIC_PROMPT_TEMPLATE` + `JudgeScores`), data flow (`write_results`/`load_results` in Task 3, consumed by Task 4 and the CLI), report format (Task 5), error handling (server-startup skip in Task 3, retry-then-mark-unscored in Task 4, retry-then-mark-failed in Task 3), testing approach (mocked subprocess/HTTP throughout, no live calls) — all covered.
- **Placeholder scan:** No TBD/TODO markers; every step has runnable code.
- **Type consistency:** `Candidate`, `EvalConfig` (Task 1) used identically in Tasks 2-6. `results` dict shape (`candidates`/`images`/`results` with per-record `status`/`title`/`caption`/`tags`/`elapsed_seconds`/`error`/`scores`) is produced in Task 3, extended (`judge_error`) in Task 4, and consumed as-is in Task 5 and the CLI. `write_results`/`load_results` defined once in Task 3, reused (not redefined) in Task 4 and the CLI.
