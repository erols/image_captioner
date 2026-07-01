"""SQLite-backed state manifest tracking each image through pipeline stages."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("dedup", "raw", "caption", "publish")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    original_path TEXT PRIMARY KEY,
    current_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    phash TEXT,
    dedup_status TEXT NOT NULL DEFAULT 'pending',
    raw_status TEXT NOT NULL DEFAULT 'pending',
    caption_status TEXT NOT NULL DEFAULT 'pending',
    publish_status TEXT NOT NULL DEFAULT 'pending',
    title TEXT,
    caption TEXT,
    tags TEXT,
    error_message TEXT,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class ImageRecord:
    original_path: str
    current_path: str
    content_hash: str
    phash: str | None
    dedup_status: str
    raw_status: str
    caption_status: str
    publish_status: str
    title: str | None
    caption: str | None
    tags: list[str]
    error_message: str | None
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ImageRecord":
        return cls(
            original_path=row["original_path"],
            current_path=row["current_path"],
            content_hash=row["content_hash"],
            phash=row["phash"],
            dedup_status=row["dedup_status"],
            raw_status=row["raw_status"],
            caption_status=row["caption_status"],
            publish_status=row["publish_status"],
            title=row["title"],
            caption=row["caption"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            error_message=row["error_message"],
            updated_at=row["updated_at"],
        )


class Manifest:
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def register(self, original_path: Path, content_hash: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO images
                (original_path, current_path, content_hash, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (str(original_path), str(original_path), content_hash, now),
        )
        self._conn.commit()

    def get(self, original_path: str) -> ImageRecord | None:
        row = self._conn.execute(
            "SELECT * FROM images WHERE original_path = ?", (original_path,)
        ).fetchone()
        return ImageRecord.from_row(row) if row else None

    def update_stage(
        self,
        original_path: str,
        stage: str,
        status: str,
        *,
        current_path: str | None = None,
        phash: str | None = None,
        title: str | None = None,
        caption: str | None = None,
        tags: list[str] | None = None,
        error_message: str | None = None,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        fields: dict[str, object] = {
            f"{stage}_status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if current_path is not None:
            fields["current_path"] = current_path
        if phash is not None:
            fields["phash"] = phash
        if title is not None:
            fields["title"] = title
        if caption is not None:
            fields["caption"] = caption
        if tags is not None:
            fields["tags"] = json.dumps(tags)
        if error_message is not None:
            fields["error_message"] = error_message
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        self._conn.execute(
            f"UPDATE images SET {set_clause} WHERE original_path = ?",
            (*fields.values(), original_path),
        )
        self._conn.commit()

    def pending(self, stage: str) -> list[ImageRecord]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        rows = self._conn.execute(
            f"SELECT * FROM images WHERE {stage}_status = 'pending'"
        ).fetchall()
        return [ImageRecord.from_row(r) for r in rows]

    def failed(self, stage: str) -> list[ImageRecord]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        rows = self._conn.execute(
            f"SELECT * FROM images WHERE {stage}_status = 'failed'"
        ).fetchall()
        return [ImageRecord.from_row(r) for r in rows]

    def status_counts(self, stage: str) -> dict[str, int]:
        """Count records per status value for the given stage's status column."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        rows = self._conn.execute(
            f"SELECT {stage}_status AS status, COUNT(*) AS n FROM images GROUP BY {stage}_status"
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}
