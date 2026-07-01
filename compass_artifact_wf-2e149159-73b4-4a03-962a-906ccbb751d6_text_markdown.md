# Local Image Captioning on Strix Halo: Best Open-Source VLMs for Creative/Abstract Photos (mid-2026)

## TL;DR
- **Top pick: JoyCaption (Beta One), an 8B LLaVA-architecture model (Llama 3.1 8B + SigLIP2)** — purpose-built for long, descriptive, interpretive captions of *any* image style (photoreal, digital art, abstract, NSFW), uncensored, open weights, and small enough (~17GB at BF16) to run several parallel copies in your 128GB unified memory. Its creator states the goal explicitly: to perform "near or on-par with GPT4o in captioning images, while being free, unrestricted, and open." Run it via the **ROCm/PyTorch or vLLM path**, not the buggy Vulkan mmproj path.
- **Runner-up for the most ambiguous/abstract images: a large reasoning VLM — Qwen3-VL-32B (or the fast 30B-A3B MoE) or InternVL3.5-38B** — which give richer interpretation of unusual compositions and fit comfortably in your memory. **ToriiGate-0.5 (Qwen3-VL-based)** is the specialist if your "abstract" images skew toward digital art/illustration.
- **On Strix Halo the hardware is capable but the vision software stack is the real constraint:** the mmproj/CLIP vision encoder on the **Vulkan/RADV** backend in llama.cpp has documented correctness and crash bugs on AMD GPUs; use **ROCm (HIP) or CPU for the vision-encoding step**, or a PyTorch/vLLM ROCm path. For batch work, **taggui** or the **"A Thousand Words"** captioner on the ROCm PyTorch path is the most practical workflow.

## Key Findings

**1. "Creative/interpretive" vs "literal" captioning is a real, model-dependent distinction.** For abstract or ambiguous images you want a model whose training biases it toward long, narrative, atmosphere-aware descriptions rather than terse object lists. The community-standard model for exactly this is **JoyCaption**, explicitly built to write "long descriptive captions." Its creator notes its descriptive mode deliberately "uses hedging words ('likely', 'probably', etc), includes extraneous details like the mood of the image" — precisely the interpretive quality you want for abstract art. **ToriiGate** is the other purpose-built art captioner. General-purpose VLMs (Qwen3-VL, InternVL3.5, Molmo) produce creative captions when prompted to, but their default style is more literal/analytical.

**2. Your 128GB unified memory is a genuine advantage — but mainly for the language-model half.** Strix Halo (Ryzen AI Max+ 395, Radeon 8060S iGPU, gfx1151) exposes a large VRAM-like pool: AMD's official spec caps Variable Graphics Memory at 96GB ("up to 96GB can be converted to VRAM through AMD Variable Graphics Memory"), while on Linux the GTT path reaches about 110GB of usable graphics memory versus the Windows 96GB cap. That lets you run 32B–72B-class VLMs, or even 235B-MoE-class models. The bottleneck is memory bandwidth — about 215 GB/s measured in practice against a 256 GB/s theoretical peak (llm-tracker.info's `rocm_bandwidth_test` measured ~212 GB/s) — which caps token generation speed, plus the immature vision-encoder software path.

**3. The software compatibility story is the crux for VLMs specifically.** Text-only LLM inference on Strix Halo is mature: community benchmarks report roughly 96.8–100.0 t/s generation on Qwen3-30B-class MoE models via Vulkan/RADV (e.g., "100.04 t/s with a separate Qwen3-30B-A3B-Instruct-2507 IQ4_XS b9467"). But *vision* models add an mmproj/CLIP encoder, and multiple llama.cpp GitHub issues show the **Vulkan mmproj path on AMD RADV is buggy** — degraded/wrong captions on some images (#20081), mid-image crashes (#23430), and heap corruption under sustained load (#22128). ROCm (HIP) or CPU produce correct results, and ROCm is also faster for the compute-heavy prompt-processing/vision step.

### Model-by-model comparison

| Model | Params | Base arch | License | Caption style | Best for |
|---|---|---|---|---|---|
| **JoyCaption Beta One** | 8B | LLaVA (Llama 3.1 8B + SigLIP2) | Open weights, no restrictions | Long, descriptive, interpretive; tunable modes; uncensored | **Top pick** — any style incl. abstract/artistic |
| **ToriiGate 0.5 / v0.4-7B** | 7B | Qwen3-VL / Qwen2-VL | Apache 2.0 | Dense, structured, "no purple prose"; art-focused | Digital art / illustration / anime |
| **Qwen3-VL-32B (or 30B-A3B MoE)** | 32B / 30B-A3B | Qwen3-VL | Apache 2.0 | Analytical, strong reasoning, follows creative prompts well | Ambiguous/abstract scenes needing interpretation |
| **InternVL3.5-38B** | 38B | InternVL | Apache 2.0 (mostly) | Strong captioning benchmarks, detailed | Runner-up general captioner |
| **Molmo-7B / 72B; Molmo 2 (2025)** | 7B / 72B; 4B/8B | OLMo/Qwen + CLIP | Apache 2.0 | Detailed, grounded ("pointing") | Detailed factual description |
| **Florence-2 (base/large)** | 0.23B / 0.77B | Seq2seq | MIT | Terse to "more detailed caption"; fast | Lightweight fallback, bulk pre-tagging |
| **Gemma 3 vision (4/12/27B)** | 4–27B | Gemma 3 | Gemma license | General; runs on AMD LM Studio | Convenient GUI option |

**JoyCaption specifics:** Built by "fpgaminer," it is a from-scratch LLaVA-style VLM designed for captioning diffusion-training datasets. Beta One was trained on 2.4 million training samples ("I decided to double the training time to 2.4 million training samples"). It offers "equal coverage of SFW and NSFW concepts" and deliberate diversity across "digital art, photoreal, anime, furry." At its native BF16 it "needs about 17GB of VRAM for the model, so it runs comfortably on 24GB and up GPUs" — trivial for your box, where you can run multiple instances in parallel. It ships GGUF quants (Q2_K ~3.2GB up to F16 ~16.1GB) plus an mmproj vision file, an FP8 vLLM build, and MLX builds. Its default prompt is literally "Write a long descriptive caption for this image," ideal for evocative output, with tunable length and formal-vs-casual tone. The community captioning UIs (taggui, ComfyUI-JoyCaption, "A Thousand Words") all support it with a Caption Length knob.

**ToriiGate specifics:** By "Minthy," designed for "captioning of anime pictures, digital artworks and various images." v0.4-7B is Qwen2-VL-based and "finetuned with dataset of over 900k of artworks with various captions"; it is "the only opensource small-sized VLM that can handle character names well." The newest **ToriiGate-0.5** is Qwen3-VL-based and claims "state of the art level of knowledge for popular characters... up to 12.2025." Its captions are "more meaningful and dense without purple prose fillers." It supports structured JSON output, booru-tag grounding, and character recognition; Apache 2.0. Caveat: it "requires special handling" and won't work in generic chat UIs like SillyTavern/OpenWebUI.

**Qwen3-VL** (released Oct–Nov 2025) is the strongest general open VLM family in 2025–2026, spanning dense 2B/4B/8B/32B and MoE 30B-A3B/235B-A22B, all with Instruct and reasoning "Thinking" editions, Apache 2.0, 256K context. The 32B or the fast 30B-A3B MoE are the sweet spot for your box. Newer Qwen "3.5/3.6" VL models also appear in 2026 sources but require llama.cpp-compatible backends and separate mmproj files (Unsloth notes "no Qwen3.5 GGUF works in Ollama due to separate mmproj vision files").

**Molmo / Molmo 2** (Allen AI/Ai2) is fully open (weights + PixMo data), trained on highly detailed human-spoken image captions; the 72B rivals GPT-4o-class proprietary models, and Molmo 2 (Dec 2025) adds video (its 8B "exceeds the original Molmo 72 billion-parameter model on key image understanding tasks"). Good detailed describer, Apache 2.0.

**Florence-2** (Microsoft, MIT) is tiny (0.23B/0.77B) and fast, with `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>` modes. More literal than creative, but an excellent lightweight fallback and bulk pre-tagger; PromptGen fine-tunes exist for dataset captioning.

### Deployment on Strix Halo — what actually works

**Text LLM baseline is excellent:** ~96.8–100.0 t/s generation on Qwen3-30B-class MoE models via Vulkan/RADV on a Ryzen AI Max+ 395 / 128GB box, with ROCm winning on long-context prompt processing.

**Vision is the caveat (subagent-confirmed):**
- **Vulkan/RADV mmproj is fast but unreliable.** In llama.cpp issue #20081, on an AMD Radeon 780M (RADV PHOENIX), Vulkan produced a completely wrong caption — "The image displays a list of various words in Vietnamese, such as 'tinh,' 'thanh,' and 'trường,' arranged vertically against a plain white background" — where CUDA/CPU correctly described a Gazebo drone-simulation screenshot; the reporter concluded "the issue is likely with the Vulkan implementation rather than the model." Issue #23430 (Qwen3.6-27B on RX 9060 XT/RX 6600 RADV): "during image processing it processes %1 but then gets stuck," crashing in `ggml_vk_submit`/`libggml-vulkan.so`. Issue #22128 (Qwen2.5-VL-7B + mmproj on Navi 21, Mesa 25.0.7): the prompt cache "corrupts the heap," a "deterministic crash between req 300–500," fixable with `--cache-ram 0` ("Verified stable over 600 consecutive real-workload requests"). A llama.cpp discussion measured the CLIP encode step at ~6s on Vulkan (but numerically wrong) vs ~22s on ROCm (correct) vs ~30s on CPU on adjacent AMD hardware.
- **ROCm/PyTorch works in production on gfx1151.** One documented project (tinycomputers.io) batch-captioned 51,414 photos on a Strix Halo APU using BLIP (Salesforce/blip-image-captioning-large) via PyTorch 2.9.1+rocm6.3: "Caption inference takes about 0.5–0.7 seconds per image," BLIP using "roughly 2 GB of the 65.2 GB available VRAM," "Total wall time: 15.5 hours across two passes for 51,414 photos," with the `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround. A vLLM+ROCm project (hec-ovi/vllm-qwen) runs Qwen3.6-27B *with vision* on the Ryzen AI Max+ 395.
- **gfx1151 ROCm vision gotcha:** MIOpen lacks pre-compiled solver DBs for gfx1151, which can hang vision-encoder kernel search; the community workaround is to disable encoder profiling.
- **ROCm on gfx1151 is officially "Preview,"** requiring `HSA_OVERRIDE_GFX_VERSION=11.5.1` (or 11.0.0 for some PyTorch stacks); community toolboxes (kyuz0/amd-strix-halo-toolboxes) ship prebuilt Vulkan and ROCm llama.cpp containers.
- **Windows:** LM Studio's Vulkan backend supports Strix Halo (llama.cpp 2.22.1 runtime) and added vision-model support, but inherits the same Vulkan mmproj risks and runs ~20–30% slower than the Linux/RADV path; ROCm has no official Windows support for gfx1151.

### Batch captioning workflows
- **taggui** (jhc13/taggui) — cross-platform desktop app for dataset captioning; supports JoyCaption, Florence-2, CogVLM, LLaVA, WD Tagger, and more, with batch selection, prompt templates, "discourage/include" word controls, and an SDXL token counter. GPU generation is documented for NVIDIA or CPU; on AMD you'd rely on the ROCm PyTorch build.
- **"A Thousand Words"** (Civitai) — a Gradio GUI + CLI captioner supporting 20+ models (JoyCaption, Florence-2, Qwen2.5/Qwen3-VL, Moondream, Pixtral, ToriiGate, etc.) with per-model batch/queue processing and knobs like `--model joycaption --temperature 0.8 --top-k 60` for creative output and folder-level jobs.
- **ComfyUI nodes** — ComfyUI-JoyCaption (GGUF + batch Caption Tools), comfyui_toriigate, and Florence-2 auto-captioning workflows ("FACT") let you drop a directory in and export per-image `.txt` captions.
- **ToriiGate-batch** and custom `transformers`/`vllm` scripts for headless directory processing.

## Details

**Why JoyCaption for abstract images.** The core problem you described — wanting evocative interpretation, not "a red square on a blue background" — is precisely JoyCaption's design goal. Its dataset is built around long-form (≈200-word) descriptive captions that mention "emotions, or meaning the image was meant to convey," and it explicitly avoids the clinical evasiveness of censored models. Because it's only 8B, on your 128GB box you can run it at full BF16 (~17GB) with enormous headroom, or run several instances in parallel to speed up a large collection. Its weakness is that it's a *specialist* — it only captions; it isn't a general chat/reasoning VLM, and for genuinely puzzling abstract art a larger reasoning model may "interpret" more interestingly.

**Why a large model as backup.** For the most ambiguous images, a 32B+ reasoning-capable VLM (Qwen3-VL-32B/Thinking, or InternVL3.5-38B with its chain-of-thought captioning) can produce more layered interpretations when prompted explicitly ("Describe this image evocatively; interpret its mood and possible meaning"). Your hardware is one of the few consumer platforms that can hold these locally. The tradeoff is speed: token generation is memory-bandwidth-bound, so long captions from a 32B model will be slower than from JoyCaption 8B.

**Licensing notes.** JoyCaption is released "free, open weights, no restrictions" (community-permissive, though not a formal SPDX license — verify before commercial redistribution). ToriiGate, Qwen3-VL, Molmo, InternVL3.5, and Florence-2 are Apache-2.0 or MIT (very permissive). Gemma 3/PaliGemma use Google's Gemma license (permissive but with a use policy). Moondream 3 uses BSL 1.1 (source-available, restrictions).

## Recommendations

**Stage 1 — Get a correct, fast pipeline running (do this first).**
- Install Linux (Ubuntu 24.04 / Fedora 42 recommended), set BIOS UMA/GTT so the iGPU can use up to ~96GB (Windows) or ~110GB (Linux GTT), and set `HSA_OVERRIDE_GFX_VERSION`. Use the **kyuz0/amd-strix-halo-toolboxes** ROCm container to avoid build pain.
- Install **JoyCaption Beta One** and run it via **transformers/PyTorch on ROCm** (or vLLM on ROCm). Do **not** offload the mmproj/vision encoder to Vulkan.
- Wrap it with **taggui** or **"A Thousand Words"** for batch folder processing, using JoyCaption's "long descriptive" mode with temperature ~0.7–0.8 for creative variety. Expect roughly BLIP-class-to-slower per-image throughput (BLIP hit 0.5–0.7 s/image on this hardware; an 8B VLM writing long captions will be slower, so plan on parallel instances for a large collection).

**Stage 2 — Add an interpretive large model for hard images.**
- Pull **Qwen3-VL-32B** (or 30B-A3B MoE) GGUF + matching mmproj, or **InternVL3.5-38B**. Run text on GPU; keep the vision encoder on ROCm/CPU. Prompt explicitly for evocative/interpretive captions. Route only your most abstract images here to save time.

**Stage 3 — Lightweight fallback / bulk pre-pass.**
- Keep **Florence-2-large** (MIT, <1GB) for instant `<MORE_DETAILED_CAPTION>` on the whole collection as a first pass or sanity check; it's fast even on CPU.

**Benchmarks/thresholds that would change the recommendation:**
- If a llama.cpp release fixes the **Vulkan mmproj bugs** (#20081/#23430/#22128) for gfx1151, switch the vision step to Vulkan for a ~3–5× encode speedup.
- If ROCm moves gfx1151 from "Preview" to full support and ships MIOpen solver DBs, prefer ROCm end-to-end.
- If your collection is mostly anime/illustration rather than photos, promote **ToriiGate-0.5** to top pick.
- If throughput is too slow, drop to JoyCaption at Q5/Q6 GGUF or run multiple parallel instances rather than moving to a bigger model.

## Caveats
- **Vision-on-Strix-Halo evidence is partly adjacent, not all gfx1151-specific.** The strongest Vulkan-mmproj bug reports are on related AMD RADV GPUs (Radeon 780M/PHOENIX, RX 9060/6600, Navi 21) that share the same llama.cpp Vulkan code path; the concrete "6s Vulkan vs 22s ROCm vs 30s CPU" encode timings are from adjacent AMD hardware, not a confirmed Strix Halo box. Treat these as strong indicators and test on your own machine. Verified positive gfx1151 vision evidence exists via the ROCm/PyTorch (BLIP, 51k photos) and vLLM (Qwen3.6-27B vision) paths.
- **Fast-moving software.** gfx1151 support in ROCm, llama.cpp, and LM Studio changed rapidly through 2025–2026; firmware/kernel choices matter (community reports that `linux-firmware-20251125` breaks ROCm on Strix Halo — use 20240318 or 20260110+). Kernels older than 6.18.4 also have a gfx1151 stability bug. Verify current versions.
- **JoyCaption licensing is "open weights, no restrictions" but not a standard license file** — confirm terms before commercial redistribution.
- **Some 2026 model names in sources (Qwen "3.5/3.6" VL, Gemma 4, Molmo 2) are newer and less battle-tested for this specific workflow.** AMD's Gemma 3 vision figures ("up to 4.6x faster in Google Gemma 3 4b and up to 6x faster in Google Gemma 3 12b") are vendor benchmarks on a 64GB ASUS ROG Flow Z13 laptop, not absolute throughput on your 128GB desktop.
- **"Uncensored" models (JoyCaption, ToriiGate) will describe adult/sensitive content literally** — appropriate for a personal collection, but be aware if outputs are shared.