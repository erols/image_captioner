# Model Evaluation Harness — Design (Sub-project 2 of 3)

**Status:** Approved for planning
**Date:** 2026-07-02

## Context

This is the second of the three sub-projects that make up the full image-captioner app:

1. **Core pipeline** — dedup → RAW convert → resize → caption → rename + MD output. Complete, merged to `main`, validated end-to-end.
2. **Model evaluation harness (this doc)** — run multiple candidate VLMs over a sample set, score captions via an LLM judge, compare cost/quality/speed.
3. **Categorization/organization pass** — a later pass over generated MD files to enrich tags/cross-links.

**Purpose driving the design:** the core pipeline already works, but it's only ever been run against `Qwen3-VL-8B-Instruct`. Caption quality is paramount (per CLAUDE.md), so before settling on a default model this harness needs to produce a repeatable, automated, apples-to-apples comparison of candidate VLMs — including ones added later, not just the two currently on disk.

**Current model inventory** (`/mnt/models/hf_ggufs/vlm/`): `Qwen3-VL-8B-Instruct-BF16` and `JoyCaption Beta One` both have a matching mmproj file present and are confirmed working end-to-end. `Llama-3.2-11B-Vision-Instruct` is `mllama` architecture, which llama.cpp's multimodal support does not cover — excluded. `moondream2` has no mmproj file present locally — excluded until one is sourced. The harness's model list is config-driven, so adding a working candidate later (moondream2, or anything new) is a config change, not a code change.

## Architecture

A new `evaluate` CLI subcommand added to the existing `image_captioner` package, backed by a new `evaluation/` module. It reuses the core pipeline's existing building blocks — `vlm_client.request_caption`, `resize.resize_for_vlm`, and `config.DEFAULT_PROMPT` — so what gets judged is exactly what production would generate for that model, not an eval-tuned prompt.

Configuration lives in a new `eval.toml`:
- `candidates`: list of `{name, model_path, mmproj_path}` — the VLMs to compare.
- `image_dir`: folder of images to caption (points at `test_pics` or similar; no fixed set — add images anytime, the harness just processes whatever's there at run time).
- `judge_model`: OpenRouter model id used as the scoring judge (API key read from `OPENROUTER_API_KEY` env var).
- `output_report`: path for the generated markdown report.
- `port`: local port to run each candidate's `llama-server` on.

Because `llama-server` can only hold one model in memory at a time, the harness owns the full lifecycle: for each candidate in turn, it launches `llama-server` (reusing `scripts/run_llama_server.sh`), waits for `/health`, captions every image in `image_dir` while timing each call, then stops the server before moving to the next candidate. Once every candidate has captioned every image, a second pass sends each (image, caption) pair to the OpenRouter judge for scoring, and a report is generated from the combined results.

### Components

- **`evaluation/config.py`** — `EvalConfig`, loaded from `eval.toml` (mirrors `config.PipelineConfig`'s `from_toml` pattern).
- **`evaluation/server_manager.py`** — starts/stops the `llama-server` subprocess per candidate; polls `/health` until ready or a timeout is hit.
- **`evaluation/runner.py`** — orchestrates the caption phase: for each candidate × image, resize + `request_caption` + record wall-clock seconds. Writes results incrementally to `eval-results.json` as it goes (not just at the end), so a crash partway through doesn't lose already-captioned work.
- **`evaluation/judge.py`** — OpenRouter client, same OpenAI-style chat-completions shape as `vlm_client` (base64 image + rubric prompt in one message), returns per-image scores.
- **`evaluation/report.py`** — reads the JSON results, computes per-model averages, writes the markdown report.
- **CLI**: `image_captioner evaluate --config eval.toml`. A `--from-captions eval-results.json` flag skips the (expensive, local) captioning phase and re-runs only the judge + report phases against already-captured captions — useful when iterating on the judge rubric.

### Scoring rubric

Per (image, model) pair, the judge returns four 1–10 scores plus brief reasoning:
- `accuracy` — does the caption match what's actually in the image
- `descriptiveness` — level of detail
- `evocativeness` — quality of the descriptive/interpretive language
- `mood_fit` — how well the caption captures atmosphere/mood in a way that would let an LLM later match this image to a music track's vibe (the stated end use of this whole app)

`accuracy` + `descriptiveness` + `evocativeness` are reported individually and as an averaged "quality" score; `mood_fit` is kept separate, since a caption can be technically accurate and detailed but emotionally flat, or vice versa.

## Data flow

```
eval.toml
  → EvalConfig
  → caption phase: for each candidate → start server → for each image: resize + caption + time it → stop server
  → eval-results.json (captions + timings)
  → judge phase: for each image × model → OpenRouter call → scores merged into the same JSON
  → report phase → eval-report.md
```

## Report format

Markdown, written to `output_report`: a summary table up top (one row per model — avg accuracy/descriptiveness/evocativeness/quality/mood_fit, avg seconds/image, image count), followed by per-image detail sections (each model's caption, scores, and judge reasoning for that image).

## Error handling

- A candidate's server fails to start or never passes `/health` within the timeout → skip that candidate, log a warning, continue with the rest.
- A single image's caption call fails after retry/backoff (same retry pattern as `caption.py`) → recorded as failed for that model/image, excluded from that model's averages, noted in the report.
- A judge call fails after retry/backoff → that image/model pair is marked "unscored" in the report rather than aborting the run.
- No SQLite manifest here — this is an occasional evaluation tool over a small image set, not a resumable production pipeline; the incrementally-written `eval-results.json` is the only persistence, and it doubles as the resume point via `--from-captions`.

## Testing

- Unit: `EvalConfig` TOML parsing; `server_manager` start/stop/health-check logic against a mocked subprocess; `judge.py` request/response parsing against mocked HTTP (valid response, malformed JSON, missing keys); `report.py` markdown generation from a canned results JSON (summary table math, per-image sections).
- No real `llama-server` or OpenRouter calls in the automated test suite — matches the existing pipeline's approach of skipping/mocking live-model integration.

## Out of scope for this sub-project

- Automatically sourcing new candidate models or mmproj files (e.g. downloading moondream2's mmproj) — config just references whatever's already on disk.
- Any comparative/pairwise judging (asking the judge to rank models against each other for the same image) — each (image, model) pair is scored independently, which keeps scores comparable across future runs as new candidates are added, not just within a single run.
- Tag enrichment or categorization of the eval results themselves — sub-project #3's concern, applied to the production pipeline's output, not this harness's report.
- A GUI — CLI only, per CLAUDE.md.
