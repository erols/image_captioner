from pathlib import Path

import pytest

from image_captioner.manifest import Manifest


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    m = Manifest(tmp_path / "manifest.sqlite3")
    yield m
    m.close()


def test_register_then_get(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")

    record = manifest.get("/photos/a.jpg")

    assert record is not None
    assert record.current_path == "/photos/a.jpg"
    assert record.content_hash == "hash1"
    assert record.dedup_status == "pending"
    assert record.raw_status == "pending"
    assert record.caption_status == "pending"
    assert record.publish_status == "pending"
    assert record.tags == []


def test_register_is_idempotent(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")
    manifest.register(Path("/photos/a.jpg"), "hash1")

    manifest.update_stage("/photos/a.jpg", "dedup", "done")
    manifest.register(Path("/photos/a.jpg"), "hash1")  # must not reset status

    record = manifest.get("/photos/a.jpg")
    assert record.dedup_status == "done"


def test_update_stage_sets_status_and_fields(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")

    manifest.update_stage(
        "/photos/a.jpg",
        "caption",
        "done",
        title="A Title",
        caption="A caption.",
        tags=["mood", "outdoor"],
    )

    record = manifest.get("/photos/a.jpg")
    assert record.caption_status == "done"
    assert record.title == "A Title"
    assert record.caption == "A caption."
    assert record.tags == ["mood", "outdoor"]


def test_update_stage_rejects_unknown_stage(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")
    with pytest.raises(ValueError):
        manifest.update_stage("/photos/a.jpg", "bogus", "done")


def test_pending_and_failed_queries(manifest: Manifest) -> None:
    manifest.register(Path("/photos/a.jpg"), "hash1")
    manifest.register(Path("/photos/b.jpg"), "hash2")
    manifest.update_stage("/photos/a.jpg", "caption", "failed", error_message="boom")

    pending = {r.original_path for r in manifest.pending("caption")}
    failed = {r.original_path for r in manifest.failed("caption")}

    assert pending == {"/photos/b.jpg"}
    assert failed == {"/photos/a.jpg"}
