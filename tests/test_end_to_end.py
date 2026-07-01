from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from image_captioner.cli import main
from image_captioner.vlm_client import CaptionResult, VLMResponseError
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


def test_run_exits_nonzero_when_a_stage_has_failed_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_solid_image(input_dir / "photo.jpg", (400, 300), (10, 20, 30))
    config_path = _write_config(tmp_path)

    runner = CliRunner()
    with patch(
        "image_captioner.caption.time.sleep"
    ), patch(
        "image_captioner.caption.request_caption",
        side_effect=VLMResponseError("permanently broken"),
    ):
        result = runner.invoke(main, ["--config", str(config_path), "run"])

    assert result.exit_code != 0
    assert "caption: 0 done, 1 failed" in result.output
