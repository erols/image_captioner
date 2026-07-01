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


if __name__ == "__main__":
    main()
