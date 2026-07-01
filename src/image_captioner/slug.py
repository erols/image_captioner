"""Filename slug generation with collision-safe hashing."""
from __future__ import annotations

import re


def slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "untitled"


def build_filename_stem(title: str, content_hash: str, hash_len: int = 6) -> str:
    return f"{slugify(title)}-{content_hash[:hash_len]}"
