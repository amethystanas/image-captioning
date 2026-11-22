import json
from pathlib import Path

import pytest

from clipcap.evaluation.report import ReportError, load_report, results_table


def _report(path: Path, *, strategy: str = "beam", **scores: float) -> Path:
    content = {
        "split": "test",
        "num_images": 2,
        "scores": {
            "bleu1": 0.6,
            "bleu2": 0.4,
            "bleu3": 0.3,
            "bleu4": 0.2,
            "rouge_l": 0.5,
            **scores,
        },
        "generation": {"strategy": strategy, "beam_size": 5, "top_p": 0.9},
        "captions": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_table_has_one_row_per_report_named_after_its_run(tmp_path: Path) -> None:
    first = _report(tmp_path / "mlp_frozen" / "eval.json", cider=0.6268)
    second = _report(
        tmp_path / "mlp_finetune" / "eval.json", strategy="greedy", cider=0.7, bleu4=0.25
    )
    lines = results_table([first, second]).splitlines()
    assert lines[0] == (
        "| Run | Split | Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |"
    )
    assert lines[1] == "|---|---|---|---:|---:|---:|---:|---:|---:|"
    assert lines[2].startswith("| mlp_frozen | test | beam (5) |")
    assert lines[3].startswith("| mlp_finetune | test | greedy |")
    assert "62.68" in lines[2]
    assert "**70.00**" in lines[3]
    assert "**25.00**" in lines[3]
    assert "**20.00**" not in lines[2]
    assert len(lines) == 4


def test_ties_are_all_bold_and_a_single_row_has_no_bold(tmp_path: Path) -> None:
    first = _report(tmp_path / "a" / "eval.json", cider=0.5)
    second = _report(tmp_path / "b" / "eval.json", cider=0.5)
    assert results_table([first, second]).count("**50.00**") == 4  # rouge_l and cider, twice
    assert "**" not in results_table([first])


def test_optional_metrics_and_custom_labels(tmp_path: Path) -> None:
    with_meteor = _report(tmp_path / "a" / "eval.json", cider=0.5, meteor=0.2)
    without = _report(tmp_path / "b" / "eval.json", strategy="nucleus", cider=0.4)
    table = results_table([with_meteor, without], labels=["With METEOR", "Sampling"])
    lines = table.splitlines()
    assert "METEOR" in lines[0]
    assert lines[2].startswith("| With METEOR |")
    assert lines[3].startswith("| Sampling | test | nucleus (p=0.9) |")
    assert "| n/a |" in lines[3]


def test_invalid_requests_are_rejected(tmp_path: Path) -> None:
    report = _report(tmp_path / "a" / "eval.json", cider=0.5)
    with pytest.raises(ValueError, match="no evaluation report given"):
        results_table([])
    with pytest.raises(ValueError, match=r"2 label\(s\) for 1 report"):
        results_table([report], labels=["a", "b"])
    with pytest.raises(FileNotFoundError, match="evaluation report not found"):
        load_report(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "content", ["not json", "[]", '{"scores": {}}', '{"scores": {"bleu4": 1}}']
)
def test_files_that_are_not_reports_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReportError, match="is not an evaluation report"):
        load_report(path)
