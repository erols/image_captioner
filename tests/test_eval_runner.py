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
