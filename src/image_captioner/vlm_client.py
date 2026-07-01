"""HTTP client for the local llama-server vision+JSON captioning endpoint."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import requests


class VLMResponseError(Exception):
    """Raised when the VLM response can't be parsed into the expected shape."""


@dataclass
class CaptionResult:
    title: str
    caption: str
    tags: list[str]


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def parse_caption_json(content: str) -> CaptionResult:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise VLMResponseError(f"no JSON object found in content: {content!r}")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VLMResponseError(f"invalid JSON in content: {content!r}") from exc
    for key in ("title", "caption", "tags"):
        if key not in data:
            raise VLMResponseError(f"missing key {key!r} in VLM JSON: {data!r}")
    tags = data["tags"]
    if not isinstance(tags, list):
        raise VLMResponseError(f"'tags' must be a list, got: {tags!r}")
    return CaptionResult(
        title=str(data["title"]), caption=str(data["caption"]), tags=[str(t) for t in tags]
    )


def request_caption(
    endpoint: str, image_path: Path, prompt: str, timeout: float = 120.0
) -> CaptionResult:
    b64 = _encode_image(image_path)
    payload = {
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
        "temperature": 0.7,
    }
    response = requests.post(
        f"{endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout
    )
    response.raise_for_status()
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise VLMResponseError(f"unexpected response shape: {body}") from exc
    return parse_caption_json(content)
