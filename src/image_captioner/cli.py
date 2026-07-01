"""Command-line entrypoint for the image-captioner pipeline."""
from __future__ import annotations

import sys
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
from image_captioner.publish import run_publish
from image_captioner.raw_convert import run_convert_raw


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
def dedup(config: PipelineConfig) -> None:
    """Find and archive near-duplicate images."""
    if _do_dedup(config):
        sys.exit(1)


@main.command(name="convert-raw")
@click.pass_obj
def convert_raw_cmd(config: PipelineConfig) -> None:
    """Convert RAW originals to JPEG and archive the originals."""
    if _do_convert_raw(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def caption(config: PipelineConfig) -> None:
    """Caption each pending, raw-converted image via the local VLM."""
    if _do_caption(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def publish(config: PipelineConfig) -> None:
    """Rename images and write OKF markdown into the output bundle."""
    if _do_publish(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def run(config: PipelineConfig) -> None:
    """Run all stages in order: dedup, convert-raw, caption, publish."""
    n_failed = 0
    n_failed += _do_dedup(config)
    n_failed += _do_convert_raw(config)
    n_failed += _do_caption(config)
    n_failed += _do_publish(config)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
