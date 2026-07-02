"""OpenRouter-backed LLM judge for scoring generated captions."""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from image_captioner.evaluation.config import EvalConfig
from image_captioner.evaluation.runner import write_results
from image_captioner.resize import resize_for_vlm

logger = logging.getLogger(__name__)

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

RUBRIC_PROMPT_TEMPLATE = (
    "You are scoring an AI-generated image caption for quality. Look at the "
    "image and the caption below, then score the caption on four axes, each "
    "from 1 (poor) to 10 (excellent):\n"
    "- accuracy: does the caption match what is actually in the image\n"
    "- descriptiveness: level of detail\n"
    "- evocativeness: quality of the descriptive/interpretive language\n"
    "- mood_fit: how well the caption captures the atmosphere/mood of the "
    "image, in a way that would help match this image to a piece of music "
    "by its vibe\n\n"
    "Caption title: {title}\n"
    "Caption body: {caption}\n\n"
    "Respond ONLY with JSON in this exact shape: "
    '{{"accuracy": <1-10>, "descriptiveness": <1-10>, "evocativeness": <1-10>, '
    '"mood_fit": <1-10>, "reasoning": "<one or two sentences>"}}'
)


class JudgeResponseError(Exception):
    """Raised when the judge response can't be parsed into the expected shape."""


@dataclass
class JudgeScores:
    accuracy: int
    descriptiveness: int
    evocativeness: int
    mood_fit: int
    reasoning: str


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def parse_judge_json(content: str) -> JudgeScores:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeResponseError(f"no JSON object found in content: {content!r}")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"invalid JSON in content: {content!r}") from exc
    for key in ("accuracy", "descriptiveness", "evocativeness", "mood_fit", "reasoning"):
        if key not in data:
            raise JudgeResponseError(f"missing key {key!r} in judge JSON: {data!r}")
    try:
        accuracy = int(data["accuracy"])
        descriptiveness = int(data["descriptiveness"])
        evocativeness = int(data["evocativeness"])
        mood_fit = int(data["mood_fit"])
        reasoning = str(data["reasoning"])
    except (ValueError, TypeError) as exc:
        raise JudgeResponseError(f"invalid score value in judge JSON: {data!r}") from exc
    for score_name, score_value in (
        ("accuracy", accuracy),
        ("descriptiveness", descriptiveness),
        ("evocativeness", evocativeness),
        ("mood_fit", mood_fit),
    ):
        if not 1 <= score_value <= 10:
            raise JudgeResponseError(
                f"score {score_name!r} out of range 1-10: {score_value!r} in judge JSON: {data!r}"
            )
    return JudgeScores(
        accuracy=accuracy,
        descriptiveness=descriptiveness,
        evocativeness=evocativeness,
        mood_fit=mood_fit,
        reasoning=reasoning,
    )


def request_judge_score(
    judge_model: str,
    image_path: Path,
    title: str,
    caption: str,
    max_dim: int = 1568,
    quality: int = 92,
    timeout: float = 120.0,
) -> JudgeScores:
    try:
        api_key = os.environ["OPENROUTER_API_KEY"]
    except KeyError as exc:
        raise JudgeResponseError("OPENROUTER_API_KEY environment variable is not set") from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        resized_path = Path(tmp_dir) / "resized.jpg"
        resize_for_vlm(image_path, resized_path, max_dim=max_dim, quality=quality)
        b64 = _encode_image(resized_path)
    prompt = RUBRIC_PROMPT_TEMPLATE.format(title=title, caption=caption)
    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(OPENROUTER_ENDPOINT, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise JudgeResponseError(f"unexpected response shape: {body}") from exc
    return parse_judge_json(content)


def run_judging(config: EvalConfig, results: dict) -> dict:
    for candidate in config.candidates:
        candidate_results = results["results"].get(candidate.name, {})
        for image_key, record in candidate_results.items():
            if record["status"] != "done" or record.get("scores") is not None:
                continue

            image_path = config.image_dir / image_key
            last_error: Exception | None = None
            for attempt in range(config.max_retries + 1):
                try:
                    scores = request_judge_score(
                        config.judge_model,
                        image_path,
                        record["title"],
                        record["caption"],
                        max_dim=config.resize_max_dim,
                        quality=config.resize_jpeg_quality,
                    )
                    record["scores"] = {
                        "accuracy": scores.accuracy,
                        "descriptiveness": scores.descriptiveness,
                        "evocativeness": scores.evocativeness,
                        "mood_fit": scores.mood_fit,
                        "reasoning": scores.reasoning,
                    }
                    record["judge_error"] = None
                    last_error = None
                    break
                except (JudgeResponseError, OSError, requests.RequestException) as exc:
                    last_error = exc
                    logger.warning(
                        "judge attempt %d/%d failed for %s/%s: %s",
                        attempt + 1,
                        config.max_retries + 1,
                        candidate.name,
                        image_key,
                        exc,
                    )
                    if attempt < config.max_retries:
                        time.sleep(2**attempt)

            if last_error is not None:
                record["scores"] = None
                record["judge_error"] = str(last_error)

            write_results(config.results_path, results)

    return results
