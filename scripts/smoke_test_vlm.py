#!/usr/bin/env python3
"""
De-risk smoke test for llama-server vision + JSON captioning.

Confirms that a running llama-server instance can:
  1. accept an image via the OpenAI-compatible vision chat API, and
  2. return content that contains a parseable {"title", "caption", "tags"}
     JSON object.

Deliberately standalone (no dependency on the image_captioner package) so it
can run the moment the server is built, before the rest of the pipeline
exists.

Usage: python3 scripts/smoke_test_vlm.py IMAGE_PATH [ENDPOINT]
"""
import base64
import json
import sys
from pathlib import Path

import requests

PROMPT = (
    "Write a long, evocative, descriptive caption for this image, capturing "
    "mood and meaning as well as literal content. Respond ONLY with JSON in "
    "this exact shape: "
    '{"title": "short title", "caption": "long descriptive caption", '
    '"tags": ["tag1", "tag2", "tag3"]}'
)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = Path(sys.argv[1])
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"

    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
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
        f"{endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=120
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print("--- raw model output ---")
    print(content)

    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        print("\nFAIL: no JSON object found in output")
        sys.exit(1)
    data = json.loads(content[start : end + 1])
    for key in ("title", "caption", "tags"):
        assert key in data, f"missing key: {key}"
    print("\n--- parsed ---")
    print(f"Title:   {data['title']}")
    print(f"Caption: {data['caption']}")
    print(f"Tags:    {data['tags']}")
    print("\nPASS")


if __name__ == "__main__":
    main()
