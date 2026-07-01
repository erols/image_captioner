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
