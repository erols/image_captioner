from datetime import datetime, timezone

import yaml

from image_captioner.okf import build_okf_document


def test_build_okf_document_has_valid_frontmatter_and_body() -> None:
    doc = build_okf_document(
        title="Quiet Harbor at Dusk",
        caption="A quiet harbor at dusk. Boats rest at anchor under a fading sky.",
        tags=["calm", "harbor", "dusk"],
        image_relative_path="quiet-harbor-at-dusk-a1b2c3.jpg",
        timestamp=datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert doc.startswith("---\n")
    _, frontmatter_raw, body = doc.split("---\n", 2)
    frontmatter = yaml.safe_load(frontmatter_raw)

    assert frontmatter["type"] == "Image Caption"
    assert frontmatter["title"] == "Quiet Harbor at Dusk"
    assert frontmatter["description"] == "A quiet harbor at dusk."
    assert frontmatter["resource"] == "quiet-harbor-at-dusk-a1b2c3.jpg"
    assert frontmatter["tags"] == ["calm", "harbor", "dusk"]
    assert frontmatter["timestamp"] == "2026-07-01T12:00:00+00:00"

    assert "![Quiet Harbor at Dusk](quiet-harbor-at-dusk-a1b2c3.jpg)" in body
    assert "Boats rest at anchor under a fading sky." in body
