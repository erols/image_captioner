# tests/test_report.py
from pathlib import Path

from image_captioner.evaluation.report import (
    compute_model_summary,
    render_report,
    write_report,
)


def _sample_results() -> dict:
    return {
        "candidates": ["cand-a"],
        "images": ["photo.jpg", "unscored.jpg"],
        "results": {
            "cand-a": {
                "photo.jpg": {
                    "status": "done",
                    "title": "Whispers of Dawn",
                    "caption": "A quiet lake at sunrise.",
                    "tags": ["lake", "dawn"],
                    "elapsed_seconds": 2.5,
                    "error": None,
                    "scores": {
                        "accuracy": 8,
                        "descriptiveness": 7,
                        "evocativeness": 9,
                        "mood_fit": 6,
                        "reasoning": "Evocative and accurate.",
                    },
                },
                "unscored.jpg": {
                    "status": "done",
                    "title": "Other",
                    "caption": "Something else.",
                    "tags": [],
                    "elapsed_seconds": 1.0,
                    "error": None,
                    "scores": None,
                    "judge_error": "OPENROUTER_API_KEY environment variable is not set",
                },
            }
        },
    }


def test_compute_model_summary_averages_scores_and_counts() -> None:
    results = _sample_results()
    summary = compute_model_summary("cand-a", results["results"]["cand-a"])

    assert summary["n_images"] == 2
    assert summary["n_captioned"] == 2
    assert summary["n_scored"] == 1
    assert summary["avg_accuracy"] == 8.0
    assert summary["avg_quality"] == (8 + 7 + 9) / 3
    assert summary["avg_mood_fit"] == 6.0
    assert summary["avg_seconds"] == (2.5 + 1.0) / 2


def test_compute_model_summary_handles_no_scored_images() -> None:
    candidate_results = {
        "photo.jpg": {
            "status": "failed",
            "title": None,
            "caption": None,
            "tags": None,
            "elapsed_seconds": None,
            "error": "boom",
            "scores": None,
        },
    }
    summary = compute_model_summary("cand-a", candidate_results)

    assert summary["n_captioned"] == 0
    assert summary["n_scored"] == 0
    assert summary["avg_accuracy"] is None
    assert summary["avg_seconds"] is None


def test_render_report_includes_summary_table_and_per_image_detail() -> None:
    report = render_report(_sample_results())

    assert "| cand-a |" in report
    assert "Whispers of Dawn" in report
    assert "accuracy=8" in report
    assert "unscored" in report
    assert "OPENROUTER_API_KEY environment variable is not set" in report


def test_write_report_writes_file(tmp_path: Path) -> None:
    output_path = tmp_path / "eval-report.md"
    write_report(_sample_results(), output_path)

    assert output_path.exists()
    assert "cand-a" in output_path.read_text()
