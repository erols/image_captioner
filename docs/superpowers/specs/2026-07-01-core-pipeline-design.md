# Core Image Captioning Pipeline — Design (Sub-project 1 of 3)

**Status:** Approved for planning
**Date:** 2026-07-01

## Context

This is the first of three sub-projects that make up the full image-captioner app:

1. **Core pipeline (this doc)** — dedup → RAW convert → resize → caption → rename + MD output, using a model-agnostic local VLM interface.
2. **Model evaluation harness** — run multiple candidate VLMs over a sample set, score captions (possibly via an LLM judge such as Opus/OpenRouter), compare cost/quality/speed. Depends on #1 existing.
3. **Categorization/organization pass** — a later pass over generated MD files to enrich tags/cross-links. Depends on #1's OKF output existing.

**Purpose driving several decisions below:** the output bundle is meant to be a resource an LLM agent can search to find images matching a mood/description (e.g., for pairing images with music tracks). This favors frontmatter-driven retrieval (tags, description, caption body) over directory-based organization, and argues for a **flat** output structure — OKF's own design treats tags/cross-links, not folders, as the categorization mechanism, and reshuffling directories later would break path-based links.

**Hardware:** Strix Halo (Ryzen AI Max+ 395), 128GB unified memory, ROCm 7.2.2 already installed. GGUF VLMs on disk at `/mnt/models/hf_ggufs/vlm`: `Llama-3.2-11B-Vision-Instruct` (Q8_0, f16), `Qwen3-VL-8B-Instruct` (BF16), `moondream2`, and `JoyCaption Beta One` (LLaVA architecture — Llama 3.1 8B + SigLIP2; download in progress as of this doc). JoyCaption is the research doc's top pick for caption quality, so it's the primary candidate for the MVP de-risk task and the eventual evaluation harness (sub-project #2) baseline.

## Architecture

A CLI tool, dependency-managed with `uv`, exposing **one subcommand per pipeline stage**. All stages share a **SQLite state manifest** (one row per source image) recording: content hash, perceptual hash, current stage, status (`pending` / `done` / `failed`), and output paths. This makes every stage idempotent and resumable — re-running a stage skips images already marked `done` for that stage, and retries `failed` ones. SQLite (not JSON) is chosen because the evaluation harness (sub-project #2) will run multiple models over the same images and needs concurrency-safe status writes.

### Pipeline stages

1. **`dedup`** — Compute a perceptual hash (phash/dhash) per image. Group near-duplicates (not just byte-identical). For each group, keep one canonical image and move the rest to `_duplicates/`. Record the decision in the manifest.
2. **`convert-raw`** — For RAW inputs (CR2, CR3, NEF, ARW, DNG), use `rawpy`/libraw to produce a JPEG. Move the original RAW file to `_raw_originals/`. Standard formats (JPEG, PNG, HEIC/HEIF, TIFF) pass through unchanged.
3. **`caption`** — Resize a working copy so the longest edge is ≤ ~1568px (JPEG quality ~92); this copy is ephemeral, not kept in output. Send it to the local VLM (see Model Serving) with a single prompt requesting structured JSON: `{"title": ..., "caption": ..., "tags": [...]}`. Tags are short mood/subject keywords, generated in the same call at no extra inference cost, to seed retrieval before sub-project #3 exists. On failure or malformed JSON: retry up to 2 times with backoff; if still failing, log the error and mark the image `failed` in the manifest, then continue the batch (never halt the whole run).
4. **`publish`** — Build the final filename: a slug of the title plus a short content-hash suffix (e.g. `sunset-over-harbor-a1b2c3.jpg`) — guarantees uniqueness without needing to scan existing files for collisions. Rename the image to this. Write a sibling `.md` file (same base name) in **OKF format**:

   ```yaml
   ---
   type: Image Caption
   title: <VLM title>
   description: <first sentence of the VLM caption>
   resource: <relative path to the renamed image>
   tags: [<VLM tags>]
   timestamp: <ISO 8601 processing time>
   ---

   ![<title>](<relative path to image>)

   <full VLM caption text>
   ```

   Both files land **flat** in the output bundle root — no subfolders. This bundle is Obsidian-vault-ready as-is.

### Folder layout

```
output/                     # flat OKF bundle: image + .md pairs, no subfolders
_duplicates/                # near-duplicate images moved aside by `dedup`
_raw_originals/             # original RAW files moved aside by `convert-raw`
```

`_duplicates/` and `_raw_originals/` are siblings of the output bundle, not part of it (they're not OKF concepts).

### Model serving

`llama-server` (llama.cpp), **built from source** with `cmake -DGGML_HIP=ON` against the installed ROCm 7.2.2 — not the Vulkan/RADV path, which has documented mmproj correctness/crash bugs on AMD GPUs. This is the "self-contained script to run the VLM" the app ships with. The server exposes an OpenAI-compatible HTTP+vision API; the pipeline is a plain HTTP client against it, configured with an endpoint URL and a prompt template — swapping models means changing config, not code. This is the interface the evaluation harness (sub-project #2) will plug into.

**Unverified risk — first implementation task, not an assumption to build on top of:**
- `Llama-3.2-11B-Vision-Instruct` is `mllama` architecture; llama.cpp's multimodal (`mtmd`) support for it has a rocky history and must be confirmed working in the current `llama-server`, not just assumed from the GGUF being present.
- JoyCaption is a from-scratch LLaVA-style model (not a mainstream architecture llama.cpp ships day-one support for); its GGUF + mmproj pairing needs the same confirmation once the download completes.
- Grammar/JSON-constrained output combined with an image in the same request is not proven to work cleanly in `llama-server`; text-only JSON-constrained output is well-supported, but the vision+JSON combination needs its own check.
- **De-risk task:** build `llama-server` with HIP, load `Qwen3-VL-8B-Instruct` + its mmproj first (most likely to "just work"), send one real image, request JSON output, confirm it parses. Then repeat against JoyCaption once its download finishes, since it's the priority model for caption quality. If JSON-constrained output doesn't hold with vision input, fall back to prompting for JSON in plain text and parsing leniently (or splitting into two calls) rather than relying on server-side grammar constraints.

## Data flow summary

```
input/ (recursive)
  → dedup            → _duplicates/ (moved aside), manifest updated
  → convert-raw       → _raw_originals/ (moved aside), JPEG written, manifest updated
  → caption           → VLM JSON {title, caption, tags}, manifest updated
  → publish           → output/ (flat: renamed image + OKF .md), manifest updated
```

## Error handling

- VLM call fails or returns unparsable JSON → retry twice with backoff → mark `failed` in manifest, log, continue batch.
- Any other stage: failures on a single image are logged and marked `failed` in the manifest; the batch continues. A later re-run of the same stage picks up `failed` and `pending` images.

## Testing

- Unit: slug + collision-hash filename generation; OKF frontmatter generation and round-trip; phash duplicate grouping on sample image sets; RAW→JPEG conversion on sample files per format; manifest state transitions (pending → done / failed, resume behavior).
- Integration: `caption` stage against a live `llama-server` instance — skipped automatically if the server isn't reachable, runnable locally against real hardware for the de-risk task and ongoing development.

## Out of scope for this sub-project

- Model evaluation/comparison across multiple VLMs (sub-project #2).
- Tag enrichment, cross-linking, and organization beyond what the single VLM call produces (sub-project #3).
- Any GUI — CLI only, per CLAUDE.md.
