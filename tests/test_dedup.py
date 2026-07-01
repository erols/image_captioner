from pathlib import Path

from image_captioner.dedup import group_duplicates, run_dedup
from image_captioner.manifest import Manifest
from tests.helpers import make_solid_image, make_structured_image


def test_group_duplicates_groups_within_threshold_keeping_shortest_name() -> None:
    a = Path("photo.jpg")
    a_copy = Path("photo_copy.jpg")
    b = Path("other.jpg")
    # a and a_copy differ by one bit (within threshold); b differs from both
    # in every bit (far outside threshold).
    phashes = {a: "ffff000000000000", a_copy: "ffff000000000001", b: "0000ffffffffffff"}

    groups = group_duplicates([a, a_copy, b], phashes, max_distance=2)

    groups_as_sets = [set(g) for g in groups]
    assert {a, a_copy} in groups_as_sets
    assert [b] in [list(g) for g in groups if len(g) == 1]
    dup_group = next(g for g in groups if len(g) == 2)
    assert dup_group[0] == a  # shorter name kept first


def test_group_duplicates_prefers_raw_as_keeper() -> None:
    raw_file = Path("shot.cr2")
    jpeg_file = Path("shot.jpg")
    phashes = {raw_file: "0" * 16, jpeg_file: "0" * 16}

    groups = group_duplicates([raw_file, jpeg_file], phashes, max_distance=2)

    assert groups[0][0] == raw_file


def test_run_dedup_moves_duplicates_and_updates_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    duplicates_dir = tmp_path / "_duplicates"
    make_solid_image(input_dir / "sunset.jpg", (800, 600), (200, 100, 50))
    make_solid_image(input_dir / "sunset_dup.jpg", (800, 600), (200, 100, 50))
    make_structured_image(input_dir / "forest.jpg", (800, 600), (0, 255, 0), (100, 255, 100))

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        run_dedup(input_dir, duplicates_dir, manifest, max_distance=5)

        remaining = sorted(p.name for p in input_dir.iterdir())
        moved = sorted(p.name for p in duplicates_dir.iterdir())
        assert len(remaining) == 2  # one of the near-duplicates moved out
        assert len(moved) == 1

        moved_record = manifest.get(str(input_dir / moved[0]))
        assert moved_record.dedup_status == "done"
        assert moved_record.raw_status == "skipped"
        assert moved_record.caption_status == "skipped"
        assert moved_record.publish_status == "skipped"
    finally:
        manifest.close()
