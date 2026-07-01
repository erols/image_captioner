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


from image_captioner.caption import run_caption
from image_captioner.dedup import run_dedup
from image_captioner.manifest import Manifest
from image_captioner.raw_convert import run_convert_raw


@main.command()
@click.pass_obj
def dedup(config: PipelineConfig) -> None:
    """Find and archive near-duplicate images."""
    manifest = Manifest(config.manifest_path)
    try:
        run_dedup(config.input_dir, config.duplicates_dir, manifest, config.phash_max_distance)
    finally:
        manifest.close()


@main.command(name="convert-raw")
@click.pass_obj
def convert_raw_cmd(config: PipelineConfig) -> None:
    """Convert RAW originals to JPEG and archive the originals."""
    manifest = Manifest(config.manifest_path)
    try:
        run_convert_raw(config.raw_originals_dir, manifest)
    finally:
        manifest.close()


@main.command()
@click.pass_obj
def caption(config: PipelineConfig) -> None:
    """Caption each pending, raw-converted image via the local VLM."""
    manifest = Manifest(config.manifest_path)
    try:
        run_caption(config, manifest)
    finally:
        manifest.close()


if __name__ == "__main__":
    main()
