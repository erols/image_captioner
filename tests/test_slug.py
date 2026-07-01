from image_captioner.slug import build_filename_stem, slugify


def test_slugify_basic() -> None:
    assert slugify("Quiet Harbor at Dusk") == "quiet-harbor-at-dusk"


def test_slugify_strips_punctuation_and_collapses_hyphens() -> None:
    assert slugify("Wow!! It's -- a Sunset??") == "wow-it-s-a-sunset"


def test_slugify_empty_title_falls_back() -> None:
    assert slugify("   ") == "untitled"


def test_build_filename_stem_includes_hash_suffix() -> None:
    stem = build_filename_stem("Quiet Harbor", "abcdef1234567890", hash_len=6)
    assert stem == "quiet-harbor-abcdef"


def test_build_filename_stem_differs_for_different_hashes_with_same_title() -> None:
    stem_a = build_filename_stem("Quiet Harbor", "111111", hash_len=6)
    stem_b = build_filename_stem("Quiet Harbor", "222222", hash_len=6)
    assert stem_a != stem_b
