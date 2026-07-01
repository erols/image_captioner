from pathlib import Path

from image_captioner.manifest import Manifest
from image_captioner.publish import run_publish
from tests.helpers import make_solid_image


def test_publish_renames_image_and_writes_okf_markdown(tmp_path: Path) -> None:
    image_path = tmp_path / "IMG_0001.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))
    output_dir = tmp_path / "output"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(image_path, "abcdef123456")
        manifest.update_stage(str(image_path), "raw", "done")
        manifest.update_stage(
            str(image_path),
            "caption",
            "done",
            title="Quiet Harbor",
            caption="A quiet harbor at dusk.",
            tags=["calm", "harbor"],
        )

        run_publish(output_dir, manifest)

        jpg_files = list(output_dir.glob("*.jpg"))
        md_files = list(output_dir.glob("*.md"))
        assert len(jpg_files) == 1
        assert len(md_files) == 1
        assert jpg_files[0].stem == md_files[0].stem
        assert jpg_files[0].stem == "quiet-harbor-abcdef"

        content = md_files[0].read_text()
        assert "type: Image Caption" in content
        assert "A quiet harbor at dusk." in content

        record = manifest.get(str(image_path))
        assert record.publish_status == "done"
        assert not image_path.exists()  # moved, not copied
    finally:
        manifest.close()


def test_publish_skips_images_not_yet_captioned(tmp_path: Path) -> None:
    image_path = tmp_path / "IMG_0002.jpg"
    make_solid_image(image_path, (400, 300), (10, 20, 30))
    output_dir = tmp_path / "output"

    manifest = Manifest(tmp_path / "manifest.sqlite3")
    try:
        manifest.register(image_path, "abcdef123456")
        manifest.update_stage(str(image_path), "raw", "done")
        # caption_status left at 'pending'

        run_publish(output_dir, manifest)

        assert list(output_dir.glob("*")) == []
        record = manifest.get(str(image_path))
        assert record.publish_status == "pending"
        assert image_path.exists()
    finally:
        manifest.close()
