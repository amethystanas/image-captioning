"""Build a Markdown comparison table from saved evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_COLUMNS = (
    ("bleu1", "BLEU-1"),
    ("bleu2", "BLEU-2"),
    ("bleu3", "BLEU-3"),
    ("bleu4", "BLEU-4"),
    ("meteor", "METEOR"),
    ("rouge_l", "ROUGE-L"),
    ("cider", "CIDEr"),
)


class ReportError(ValueError):
    """Raised when a file is not an evaluation report."""


def load_report(path: str | Path) -> dict[str, Any]:
    """Read an evaluation report written by ``EvaluationResult.save``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation report not found: {path}")
    try:
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        _ = report["scores"]["bleu4"], report["split"], report["generation"]["strategy"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReportError(f"{path} is not an evaluation report: {exc!r}") from exc
    return report


def results_table(paths: Sequence[str | Path], *, labels: Sequence[str] | None = None) -> str:
    """Return a Markdown table with one row per report, scores multiplied by 100.

    Rows are labelled with ``labels`` or, by default, with the name of the run
    directory holding each report. Metrics absent from every report are left
    out; the best value of each column is set in bold.
    """
    if not paths:
        raise ValueError("no evaluation report given")
    if labels is not None and len(labels) != len(paths):
        raise ValueError(f"{len(labels)} label(s) for {len(paths)} report(s)")
    reports = [load_report(path) for path in paths]
    names = list(labels) if labels is not None else [Path(path).parent.name for path in paths]
    columns = [(key, title) for key, title in _COLUMNS if any(key in r["scores"] for r in reports)]
    best = {key: max(r["scores"][key] for r in reports if key in r["scores"]) for key, _ in columns}

    header = ["Run", "Split", "Decoding", *(title for _, title in columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---", "---", "---", *("---:" for _ in columns)]) + "|",
    ]
    for name, report in zip(names, reports, strict=True):
        cells = [name, report["split"], _decoding(report["generation"])]
        for key, _ in columns:
            value = report["scores"].get(key)
            if value is None:
                cells.append("n/a")
            else:
                text = f"{100 * value:.2f}"
                cells.append(f"**{text}**" if value == best[key] and len(reports) > 1 else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _decoding(generation: dict[str, Any]) -> str:
    strategy = str(generation["strategy"])
    if strategy == "beam":
        return f"beam ({generation['beam_size']})"
    if strategy == "nucleus":
        return f"nucleus (p={generation['top_p']})"
    return strategy
