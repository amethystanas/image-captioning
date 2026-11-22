import json
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from clipcap.config import ExperimentConfig, ModelConfig, TrainingConfig
from clipcap.training.tracking import (
    CONFIG_FILE,
    HISTORY_FILE,
    TENSORBOARD_DIR,
    RunDirectoryError,
    RunTracker,
    prepare_run_directory,
)
from clipcap.training.trainer import EpochMetrics
from tests.training.test_trainer import make_trainer


def _scalars(directory: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = EventAccumulator(str(directory / TENSORBOARD_DIR))
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def _config(tmp_path: Path, **model: object) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(**model),  # type: ignore[arg-type]
        training=TrainingConfig(output_dir=tmp_path / "run"),
    )


def test_run_directory_stores_the_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path, prefix_length=7)
    directory = prepare_run_directory(config)
    assert directory == tmp_path / "run"
    assert ExperimentConfig.from_yaml(directory / CONFIG_FILE) == config


def test_run_directory_can_be_reused_by_the_same_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prepare_run_directory(config)
    assert prepare_run_directory(config) == tmp_path / "run"


def test_run_directory_of_another_experiment_is_refused(tmp_path: Path) -> None:
    prepare_run_directory(_config(tmp_path, prefix_length=7))
    with pytest.raises(RunDirectoryError, match="different configuration"):
        prepare_run_directory(_config(tmp_path, prefix_length=8))


def test_tracker_writes_scalars_and_history(tmp_path: Path) -> None:
    with RunTracker(tmp_path, log_every=2) as tracker:
        for step in range(1, 6):
            tracker.log_step(step, loss=10.0 - step, learning_rate=0.1 * step)
        tracker.log_epoch(EpochMetrics(1, 2.0, 1.5, 0.01))
        tracker.log_epoch(EpochMetrics(2, 1.0, 1.25, 0.005))

    scalars = _scalars(tmp_path)
    assert scalars["train/step_loss"] == [(2, 8.0), (4, 6.0)]
    assert [step for step, _ in scalars["train/learning_rate"]] == [2, 4]
    assert scalars["loss/train"] == [(1, 2.0), (2, 1.0)]
    assert scalars["loss/val"] == [(1, 1.5), (2, 1.25)]
    assert scalars["perplexity/val"][0][1] == pytest.approx(4.4817, abs=1e-3)

    history = json.loads((tmp_path / HISTORY_FILE).read_text(encoding="utf-8"))
    assert [item["epoch"] for item in history] == [1, 2]
    assert history[1] == {
        "epoch": 2,
        "train_loss": 1.0,
        "val_loss": 1.25,
        "learning_rate": 0.005,
        "val_perplexity": pytest.approx(3.4903, abs=1e-3),
    }


def test_restored_history_is_continued_not_duplicated(tmp_path: Path) -> None:
    past = [EpochMetrics(1, 2.0, 1.5, 0.01), EpochMetrics(2, 1.0, 1.25, 0.005)]
    with RunTracker(tmp_path) as tracker:
        tracker.restore_history(past)
        tracker.log_epoch(EpochMetrics(2, 0.9, 1.2, 0.005))
        tracker.log_epoch(EpochMetrics(3, 0.8, 1.1, 0.001))
    history = json.loads((tmp_path / HISTORY_FILE).read_text(encoding="utf-8"))
    assert [(item["epoch"], item["train_loss"]) for item in history] == [
        (1, 2.0),
        (2, 0.9),
        (3, 0.8),
    ]


def test_tracker_rejects_invalid_logging_period(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="log_every must be positive"):
        RunTracker(tmp_path, log_every=0)


def test_trainer_reports_every_step_and_epoch_to_its_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with RunTracker(tmp_path / "run", log_every=1) as tracker:
        trainer = make_trainer(epochs=2, warmup_steps=2, learning_rate=1e-2)
        trainer.tracker = tracker
        state = trainer.fit()

    scalars = _scalars(tmp_path / "run")
    steps = [step for step, _ in scalars["train/step_loss"]]
    assert steps == list(range(1, state.step + 1))
    # The learning rate logged with a step is the one that step was taken with.
    assert [value for _, value in scalars["train/learning_rate"]][:3] == pytest.approx(
        [1e-2 / 3, 2e-2 / 3, 1e-2]
    )
    assert [value for _, value in scalars["loss/val"]] == pytest.approx(
        [metrics.val_loss for metrics in state.history]
    )
    history = json.loads((tmp_path / "run" / HISTORY_FILE).read_text(encoding="utf-8"))
    assert [item["val_loss"] for item in history] == [m.val_loss for m in state.history]
