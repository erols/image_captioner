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
