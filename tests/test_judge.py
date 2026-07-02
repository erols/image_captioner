import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_captioner.evaluation.config import Candidate, EvalConfig
from image_captioner.evaluation.judge import (
    JudgeResponseError,
    JudgeScores,
    parse_judge_json,
    request_judge_score,
    run_judging,
)
from image_captioner.evaluation.runner import load_results
from tests.helpers import make_solid_image


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


def test_parse_judge_json_non_numeric_score_raises() -> None:
    content = (
        '{"accuracy": "N/A", "descriptiveness": 7, "evocativeness": 9, '
        '"mood_fit": 6, "reasoning": "Vivid and accurate."}'
    )
    with pytest.raises(JudgeResponseError):
        parse_judge_json(content)


def test_parse_judge_json_out_of_range_score_raises() -> None:
    content = (
        '{"accuracy": 15, "descriptiveness": 7, "evocativeness": 9, '
        '"mood_fit": 6, "reasoning": "Vivid and accurate."}'
    )
    with pytest.raises(JudgeResponseError):
        parse_judge_json(content)


def test_request_judge_score_posts_expected_payload_and_parses_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (2000, 1500), (10, 20, 30))

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

    # The encoded image should be the resized (<=1568px default) version, not
    # the full-resolution 2000x1500 original.
    encoded = content_parts[1]["image_url"]["url"].split(",", 1)[1]
    decoded_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(decoded_bytes)) as decoded_image:
        assert decoded_image.format == "JPEG"
        assert max(decoded_image.size) <= 1568


def test_request_judge_score_resizes_non_jpeg_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PNG source should still be judged successfully: the resize step
    converts it to a genuine JPEG before it's base64-encoded and labeled
    image/jpeg in the request."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image_path = tmp_path / "photo.png"
    make_solid_image(image_path, (400, 300), (50, 60, 70))

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"accuracy": 8, "descriptiveness": 8, "evocativeness": 8, '
                        '"mood_fit": 8, "reasoning": "Fine."}'
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

    assert result.accuracy == 8
    payload = post.call_args.kwargs["json"]
    content_parts = payload["messages"][0]["content"]
    url = content_parts[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    encoded = url.split(",", 1)[1]
    decoded_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(decoded_bytes)) as decoded_image:
        assert decoded_image.format == "JPEG"


def test_request_judge_score_uses_given_max_dim_and_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image_path = tmp_path / "photo.jpg"
    make_solid_image(image_path, (2000, 1500), (10, 20, 30))

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"accuracy": 8, "descriptiveness": 8, "evocativeness": 8, '
                        '"mood_fit": 8, "reasoning": "Fine."}'
                    )
                }
            }
        ]
    }

    with patch(
        "image_captioner.evaluation.judge.requests.post", return_value=fake_response
    ) as post:
        request_judge_score(
            "anthropic/claude-opus-4-8",
            image_path,
            "A Title",
            "A caption body.",
            max_dim=100,
            quality=80,
        )

    payload = post.call_args.kwargs["json"]
    content_parts = payload["messages"][0]["content"]
    url = content_parts[1]["image_url"]["url"]
    encoded = url.split(",", 1)[1]
    decoded_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(decoded_bytes)) as decoded_image:
        assert max(decoded_image.size) <= 100


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

    mock_request.assert_called_once_with(
        config.judge_model,
        tmp_path / "photo.jpg",
        "T",
        "C",
        max_dim=config.resize_max_dim,
        quality=config.resize_jpeg_quality,
    )
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
