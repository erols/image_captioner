from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from image_captioner.cli import main


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


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
