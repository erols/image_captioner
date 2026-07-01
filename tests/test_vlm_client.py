from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_captioner.vlm_client import (
    CaptionResult,
    VLMResponseError,
    parse_caption_json,
    request_caption,
)


def test_parse_caption_json_clean() -> None:
    content = '{"title": "Quiet Dock", "caption": "A quiet dock at dawn.", "tags": ["calm", "water"]}'
    result = parse_caption_json(content)
    assert result == CaptionResult(
        title="Quiet Dock", caption="A quiet dock at dawn.", tags=["calm", "water"]
    )


def test_parse_caption_json_embedded_in_prose() -> None:
    content = (
        "Sure, here's the caption:\n```json\n"
        '{"title": "Quiet Dock", "caption": "A quiet dock.", "tags": ["calm"]}'
        "\n```"
    )
    result = parse_caption_json(content)
    assert result.title == "Quiet Dock"
    assert result.tags == ["calm"]


def test_parse_caption_json_missing_key_raises() -> None:
    with pytest.raises(VLMResponseError):
        parse_caption_json('{"title": "X", "caption": "Y"}')


def test_parse_caption_json_malformed_raises() -> None:
    with pytest.raises(VLMResponseError):
        parse_caption_json("not json at all, no braces")


def test_request_caption_posts_expected_payload_and_parses_response(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"title": "T", "caption": "C", "tags": ["a", "b"]}'
                }
            }
        ]
    }

    with patch("image_captioner.vlm_client.requests.post", return_value=fake_response) as post:
        result = request_caption("http://127.0.0.1:8080", image_path, "describe this")

    assert result == CaptionResult(title="T", caption="C", tags=["a", "b"])
    called_url = post.call_args.args[0]
    assert called_url == "http://127.0.0.1:8080/v1/chat/completions"
    payload = post.call_args.kwargs["json"]
    content_parts = payload["messages"][0]["content"]
    assert content_parts[0]["text"] == "describe this"
    assert content_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
