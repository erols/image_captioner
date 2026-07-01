"""OKF-format frontmatter and markdown document generation."""
from __future__ import annotations

from datetime import datetime, timezone

import yaml


def build_okf_document(
    title: str,
    caption: str,
    tags: list[str],
    image_relative_path: str,
    timestamp: datetime | None = None,
) -> str:
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    first_sentence = caption.split(". ")[0].rstrip(".") + "."
    frontmatter = {
        "type": "Image Caption",
        "title": title,
        "description": first_sentence,
        "resource": image_relative_path,
        "tags": tags,
        "timestamp": ts,
    }
    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    body = f"![{title}]({image_relative_path})\n\n{caption}\n"
    return f"---\n{yaml_block}---\n\n{body}"
