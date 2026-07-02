"""Markdown report generation from evaluation results."""
from __future__ import annotations

from pathlib import Path


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_model_summary(candidate_name: str, candidate_results: dict) -> dict:
    accuracy: list[float] = []
    descriptiveness: list[float] = []
    evocativeness: list[float] = []
    mood_fit: list[float] = []
    quality: list[float] = []
    elapsed: list[float] = []
    n_captioned = 0
    n_scored = 0

    for record in candidate_results.values():
        if record["status"] == "done":
            n_captioned += 1
            if record["elapsed_seconds"] is not None:
                elapsed.append(record["elapsed_seconds"])
        scores = record.get("scores")
        if scores is not None:
            n_scored += 1
            accuracy.append(scores["accuracy"])
            descriptiveness.append(scores["descriptiveness"])
            evocativeness.append(scores["evocativeness"])
            mood_fit.append(scores["mood_fit"])
            quality.append(
                (scores["accuracy"] + scores["descriptiveness"] + scores["evocativeness"]) / 3
            )

    return {
        "name": candidate_name,
        "n_images": len(candidate_results),
        "n_captioned": n_captioned,
        "n_scored": n_scored,
        "avg_accuracy": _average(accuracy),
        "avg_descriptiveness": _average(descriptiveness),
        "avg_evocativeness": _average(evocativeness),
        "avg_quality": _average(quality),
        "avg_mood_fit": _average(mood_fit),
        "avg_seconds": _average(elapsed),
    }


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def render_report(results: dict) -> str:
    lines = ["# Model Evaluation Report", ""]
    lines.append(
        "| Model | Images | Captioned | Scored | Accuracy | Descriptiveness | "
        "Evocativeness | Quality | Mood Fit | Avg sec/image |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    summaries = [
        compute_model_summary(name, results["results"][name]) for name in results["candidates"]
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['n_images']} | {s['n_captioned']} | {s['n_scored']} | "
            f"{_fmt(s['avg_accuracy'])} | {_fmt(s['avg_descriptiveness'])} | "
            f"{_fmt(s['avg_evocativeness'])} | {_fmt(s['avg_quality'])} | "
            f"{_fmt(s['avg_mood_fit'])} | {_fmt(s['avg_seconds'])} |"
        )

    lines.append("")
    lines.append("## Per-image detail")
    for image in results["images"]:
        lines.append("")
        lines.append(f"### {image}")
        for name in results["candidates"]:
            record = results["results"][name].get(image)
            if record is None:
                continue
            lines.append("")
            lines.append(f"**{name}**")
            if record["status"] != "done":
                lines.append(f"- captioning failed: {record['error']}")
                continue
            lines.append(f"- title: {record['title']}")
            lines.append(f"- caption: {record['caption']}")
            scores = record.get("scores")
            if scores is None:
                judge_error = record.get("judge_error")
                suffix = f": {judge_error}" if judge_error else ""
                lines.append(f"- unscored{suffix}")
            else:
                lines.append(
                    f"- scores: accuracy={scores['accuracy']}, "
                    f"descriptiveness={scores['descriptiveness']}, "
                    f"evocativeness={scores['evocativeness']}, "
                    f"mood_fit={scores['mood_fit']}"
                )
                lines.append(f"- reasoning: {scores['reasoning']}")

    return "\n".join(lines) + "\n"


def write_report(results: dict, output_path: Path) -> None:
    output_path.write_text(render_report(results))
