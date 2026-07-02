"""Command-line entrypoint for the image-captioner pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from image_captioner.config import PipelineConfig
from image_captioner.evaluation.config import EvalConfig
from image_captioner.evaluation.judge import run_judging
from image_captioner.evaluation.report import write_report
from image_captioner.evaluation.runner import load_results, run_captioning


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("pipeline.toml"),
    show_default=True,
)
@click.pass_context
def main(ctx: click.Context, config_path: Path) -> None:
    """Turn a folder of images into captioned, renamed OKF markdown notes."""
    ctx.obj = config_path


from image_captioner.caption import run_caption
from image_captioner.dedup import run_dedup
from image_captioner.manifest import Manifest
from image_captioner.publish import run_publish
from image_captioner.raw_convert import run_convert_raw


def _load_pipeline_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise click.ClickException(f"config file not found: {config_path}")
    return PipelineConfig.from_toml(config_path)


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
def dedup(config_path: Path) -> None:
    """Find and archive near-duplicate images."""
    config = _load_pipeline_config(config_path)
    if _do_dedup(config):
        sys.exit(1)


@main.command(name="convert-raw")
@click.pass_obj
def convert_raw_cmd(config_path: Path) -> None:
    """Convert RAW originals to JPEG and archive the originals."""
    config = _load_pipeline_config(config_path)
    if _do_convert_raw(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def caption(config_path: Path) -> None:
    """Caption each pending, raw-converted image via the local VLM."""
    config = _load_pipeline_config(config_path)
    if _do_caption(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def publish(config_path: Path) -> None:
    """Rename images and write OKF markdown into the output bundle."""
    config = _load_pipeline_config(config_path)
    if _do_publish(config):
        sys.exit(1)


@main.command()
@click.pass_obj
def run(config_path: Path) -> None:
    """Run all stages in order: dedup, convert-raw, caption, publish."""
    config = _load_pipeline_config(config_path)
    n_failed = 0
    n_failed += _do_dedup(config)
    n_failed += _do_convert_raw(config)
    n_failed += _do_caption(config)
    n_failed += _do_publish(config)
    if n_failed:
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "eval_config_path",
    type=click.Path(path_type=Path),
    default=Path("eval.toml"),
    show_default=True,
)
@click.option(
    "--from-captions",
    "captions_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Skip captioning and re-run judging + report against an existing eval-results.json.",
)
def evaluate(eval_config_path: Path, captions_path: Path | None) -> None:
    """Caption a folder of images with each candidate VLM and score them with an LLM judge."""
    if not eval_config_path.exists():
        raise click.ClickException(f"config file not found: {eval_config_path}")
    config = EvalConfig.from_toml(eval_config_path)

    if captions_path is not None:
        results = load_results(captions_path)
    else:
        results = run_captioning(config)

    results = run_judging(config, results)
    write_report(results, config.output_report)
    click.echo(f"Evaluation report written to {config.output_report}")


if __name__ == "__main__":
    main()
