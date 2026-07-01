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
