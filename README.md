# image-captioner

A CLI pipeline that turns a folder of images (including RAW and HEIC/HEIF)
into captioned, renamed images plus OKF-format markdown notes, using a local
vision-language model (VLM). It de-duplicates near-identical images,
converts RAW originals to JPEG, resizes a working copy for the model, asks
the model for a title/caption/tags, and publishes the renamed image and its
markdown note to an output folder.

Each stage records its status per image in a SQLite manifest, so re-running
a stage is safe: it skips images already `done` and retries any that
previously `failed`.

## Setup

```bash
uv sync
```

## Build and run the local VLM server

The pipeline talks to an OpenAI-compatible vision HTTP API served by
`llama-server` (from `llama.cpp`, built from source against ROCm/HIP — do
not use the Vulkan/RADV backend, it has known mmproj/vision bugs on AMD
GPUs). Build it once:

```bash
bash scripts/build_llama_server.sh
```

Then launch it against a GGUF vision model + its mmproj file. For example,
on a machine with models under `/mnt/models/hf_ggufs/vlm/`:

```bash
bash scripts/run_llama_server.sh \
  /mnt/models/hf_ggufs/vlm/unsloth_Qwen3-VL-8B-Instruct-BF16.gguf \
  /mnt/models/hf_ggufs/vlm/mmproj/qwen3vl-8b-mmproj-F16.gguf
```

This starts the server on `http://127.0.0.1:8080` by default (override with
`LLAMA_SERVER_PORT`). You can sanity-check it directly with:

```bash
python3 scripts/smoke_test_vlm.py path/to/image.jpg
```

## Configure the pipeline

Copy the example config and edit the paths for your setup:

```bash
cp pipeline.toml.example pipeline.toml
```

See `pipeline.toml.example` for a description of every field (input/output
folders, the manifest path, the VLM endpoint, resize settings, retry count,
and the duplicate-detection threshold). `base_dir` is required; all other
directory/manifest paths are resolved relative to it unless given as
absolute paths.

## Run the pipeline

Run every stage in order (dedup, convert-raw, caption, publish):

```bash
uv run image-captioner --config pipeline.toml run
```

Or run stages individually, e.g. to re-run just captioning after fixing a
config issue:

```bash
uv run image-captioner --config pipeline.toml dedup
uv run image-captioner --config pipeline.toml convert-raw
uv run image-captioner --config pipeline.toml caption
uv run image-captioner --config pipeline.toml publish
```

Each stage command prints a one-line `done`/`failed`/`skipped` summary and
exits non-zero if any image failed that stage; simply re-run the same
command to retry failed images.

## Evaluating candidate models

Before settling on a default VLM, compare candidates against a shared set of
images. Copy `eval.toml.example` to `eval.toml`, list the candidate models
you want to compare, and set `OPENROUTER_API_KEY` in your environment (the
harness uses OpenRouter as an independent LLM judge — never a candidate
model judging itself):

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run image-captioner evaluate --config eval.toml
```

The harness starts `llama-server` for each candidate in turn (reusing
`scripts/run_llama_server.sh`), captions every image in `image_dir`, then
scores every caption with the judge model on four axes: accuracy,
descriptiveness, evocativeness, and mood fit. Results are written
incrementally to `eval-results.json` as the run progresses, and a markdown
comparison report is written to `output_report` (`eval-report.md` by
default) when it finishes.

If you only want to re-run the judge and report — for example, after
tweaking the rubric — skip the expensive local captioning phase:

```bash
uv run image-captioner evaluate --config eval.toml --from-captions eval-results.json
```
