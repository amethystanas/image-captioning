"""Each command of the CLI, run offline with tiny stand-in models."""

import json
from pathlib import Path

import pytest

from clipcap import __version__
from clipcap.cli import main
from clipcap.config import ExperimentConfig
from clipcap.pipeline import Components, PipelineError, run_caption, run_evaluate


def _run(components: Components, *arguments: object) -> int:
    return main([str(argument) for argument in arguments], components)


def test_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == f"clipcap {__version__}"

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    output = capsys.readouterr().out
    for command in ("download", "extract", "train", "evaluate", "caption"):
        assert command in output


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["train"],
        ["unknown", "--config", "c.yaml"],
        ["evaluate", "--config", "c", "--split", "x"],
    ],
)
def test_invalid_command_lines_exit_with_usage_error(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)
    assert exit_info.value.code == 2


def test_download_reports_an_existing_dataset(
    components: Components, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(components, "download", "--config", config_path) == 0
    assert "'train': 4, 'val': 2, 'test': 2" in capsys.readouterr().out


def test_extract_caches_one_file_per_split(
    components: Components, config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(components, "extract", "--config", config_path, "--batch-size", 3) == 0
    assert sorted(path.name for path in (tmp_path / "features").iterdir()) == [
        "test.pt",
        "train.pt",
        "val.pt",
    ]
    output = capsys.readouterr().out
    assert "train: 4 embeddings of dimension 6" in output
    assert "test: 2 embeddings of dimension 6" in output


def test_train_writes_checkpoints_config_history_and_curves(
    components: Components, config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(components, "extract", "--config", config_path)
    assert _run(components, "train", "--config", config_path) == 0
    run = tmp_path / "run"
    assert {"best.pt", "last.pt", "config.yaml", "history.json", "tensorboard"} <= {
        path.name for path in run.iterdir()
    }
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert [item["epoch"] for item in history] == [1, 2]
    assert history[1]["train_loss"] < history[0]["train_loss"]
    assert "trained 2 epoch(s); best validation loss" in capsys.readouterr().out


def test_train_refuses_to_overwrite_a_run_unless_resuming(
    components: Components, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(components, "extract", "--config", config_path)
    _run(components, "train", "--config", config_path)
    capsys.readouterr()

    assert _run(components, "train", "--config", config_path) == 2
    assert "pass --resume" in capsys.readouterr().err

    assert _run(components, "train", "--config", config_path, "--resume") == 0
    assert "trained 2 epoch(s)" in capsys.readouterr().out


def test_evaluate_prints_scores_and_writes_the_report(
    components: Components, config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(components, "extract", "--config", config_path)
    _run(components, "train", "--config", config_path)
    capsys.readouterr()

    assert _run(components, "evaluate", "--config", config_path) == 0
    output = capsys.readouterr().out
    assert "test split, 2 images" in output
    assert "bleu4" in output
    assert "cider" in output
    report = json.loads((tmp_path / "run" / "eval_test_greedy.json").read_text(encoding="utf-8"))
    assert report["num_images"] == 2
    assert [item["image"] for item in report["captions"]] == ["test_0.jpg", "test_1.jpg"]

    custom = tmp_path / "reports" / "val.json"
    arguments = ("--split", "val", "--output", custom, "--checkpoint", tmp_path / "run" / "last.pt")
    assert _run(components, "evaluate", "--config", config_path, *arguments) == 0
    assert json.loads(custom.read_text(encoding="utf-8"))["split"] == "val"


def test_caption_prints_one_line_per_image(
    components: Components,
    config_path: Path,
    tiny_flickr8k: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(components, "extract", "--config", config_path)
    _run(components, "train", "--config", config_path)
    capsys.readouterr()

    images = [tiny_flickr8k / "Flicker8k_Dataset" / name for name in ("test_0.jpg", "val_1.jpg")]
    assert _run(components, "caption", "--config", config_path, *images) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split("\t")[0] for line in lines] == [str(image) for image in images]
    assert all(len(line.split("\t")) == 2 for line in lines)


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (["train"], "run 'extract' first"),
        (["evaluate"], "run 'extract' first"),
    ],
)
def test_stages_run_out_of_order_fail_cleanly(
    components: Components,
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
    message: str,
) -> None:
    assert _run(components, *command, "--config", config_path) == 2
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert message in error
    assert "Traceback" not in error


def test_evaluate_before_training_reports_the_missing_checkpoint(
    components: Components, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(components, "extract", "--config", config_path)
    assert _run(components, "evaluate", "--config", config_path) == 2
    assert "checkpoint not found" in capsys.readouterr().err


def test_user_errors_are_reported_on_one_line(
    components: Components, config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(components, "train", "--config", tmp_path / "absent.yaml") == 2
    assert "configuration file not found" in capsys.readouterr().err

    broken = tmp_path / "broken.yaml"
    broken.write_text("model: {mapper_type: cnn}", encoding="utf-8")
    assert _run(components, "train", "--config", broken) == 2
    assert "mapper_type" in capsys.readouterr().err

    nowhere = tmp_path / "nowhere.yaml"
    nowhere.write_text(f"data: {{root: {tmp_path / 'no-dataset'}}}", encoding="utf-8")
    assert _run(components, "extract", "--config", nowhere) == 2
    assert "incomplete dataset" in capsys.readouterr().err

    _run(components, "extract", "--config", config_path)
    _run(components, "train", "--config", config_path)
    capsys.readouterr()
    assert _run(components, "caption", "--config", config_path, tmp_path / "ghost.jpg") == 2
    assert "ghost.jpg" in capsys.readouterr().err


def test_pipeline_rejects_requests_the_parser_would_not_let_through(
    components: Components, config_path: Path
) -> None:
    config = ExperimentConfig.from_yaml(config_path)
    with pytest.raises(PipelineError, match="unknown split 'dev'"):
        run_evaluate(config, components, split="dev")
    with pytest.raises(PipelineError, match="no image to caption"):
        run_caption(config, [], components)


def test_captioning_a_training_image_matches_its_cached_feature_pipeline(
    components: Components, config_path: Path, tiny_flickr8k: Path, tmp_path: Path
) -> None:
    """``caption`` must embed images exactly as ``extract`` does (same normalisation)."""
    _run(components, "extract", "--config", config_path)
    _run(components, "train", "--config", config_path)
    _run(components, "evaluate", "--config", config_path)
    report = json.loads((tmp_path / "run" / "eval_test_greedy.json").read_text(encoding="utf-8"))
    from_features = {item["image"]: item["caption"] for item in report["captions"]}

    config = ExperimentConfig.from_yaml(config_path)
    images = [tiny_flickr8k / "Flicker8k_Dataset" / name for name in from_features]
    from_files = run_caption(config, images, components)
    assert list(from_files.values()) == list(from_features.values())


def test_console_script_is_installed() -> None:
    from importlib.metadata import entry_points

    scripts = {script.name: script.value for script in entry_points(group="console_scripts")}
    assert scripts["clipcap"] == "clipcap.cli:main"
