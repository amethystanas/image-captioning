"""Experiment tracking: run directory, TensorBoard scalars and loss history."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Protocol

from clipcap.config import ExperimentConfig
from clipcap.utils import get_logger

if TYPE_CHECKING:
    from clipcap.training.trainer import EpochMetrics

logger = get_logger(__name__)

CONFIG_FILE = "config.yaml"
HISTORY_FILE = "history.json"
TENSORBOARD_DIR = "tensorboard"


class RunDirectoryError(RuntimeError):
    """Raised when a run directory already belongs to a different experiment."""


class Tracker(Protocol):
    """Receiver of training metrics."""

    def log_step(self, step: int, loss: float, learning_rate: float) -> None:
        """Record the loss of one optimisation step."""
        ...

    def log_epoch(self, metrics: EpochMetrics) -> None:
        """Record the metrics of a finished epoch."""
        ...


def prepare_run_directory(config: ExperimentConfig) -> Path:
    """Create the output directory of ``config`` and store the configuration in it.

    Reusing a directory is allowed only for the very same configuration, which
    is what resuming a run does; anything else would mix the checkpoints and
    curves of two experiments.
    """
    directory = Path(config.training.output_dir)
    config_path = directory / CONFIG_FILE
    if config_path.is_file():
        if ExperimentConfig.from_yaml(config_path) != config:
            raise RunDirectoryError(
                f"{directory} already holds a run with a different configuration; "
                "choose another training.output_dir or delete it"
            )
        logger.info("reusing run directory %s", directory)
    else:
        config.to_yaml(config_path)
    return directory


class RunTracker:
    """Write TensorBoard scalars and a JSON history into a run directory."""

    def __init__(self, directory: Path, *, log_every: int = 10) -> None:
        """Open a TensorBoard writer under ``directory``.

        Step losses are written every ``log_every`` steps; epoch metrics always.
        """
        from torch.utils.tensorboard import SummaryWriter

        if log_every <= 0:
            raise ValueError("log_every must be positive")
        self.directory = directory
        self.log_every = log_every
        self._history: list[dict[str, float | int]] = []
        self._writer = SummaryWriter(log_dir=str(directory / TENSORBOARD_DIR))

    def log_step(self, step: int, loss: float, learning_rate: float) -> None:
        """Record the training loss and learning rate of step ``step``."""
        if step % self.log_every == 0:
            self._writer.add_scalar("train/step_loss", loss, step)
            self._writer.add_scalar("train/learning_rate", learning_rate, step)

    def log_epoch(self, metrics: EpochMetrics) -> None:
        """Record epoch losses and rewrite the JSON history."""
        self._writer.add_scalar("loss/train", metrics.train_loss, metrics.epoch)
        self._writer.add_scalar("loss/val", metrics.val_loss, metrics.epoch)
        self._writer.add_scalar("perplexity/val", metrics.val_perplexity, metrics.epoch)
        self._writer.flush()
        # A resumed run replays nothing: drop entries from the epoch onwards.
        self._history = [item for item in self._history if item["epoch"] < metrics.epoch]
        self._history.append({**asdict(metrics), "val_perplexity": metrics.val_perplexity})
        self._write_history()

    def restore_history(self, history: list[EpochMetrics]) -> None:
        """Seed the JSON history with the epochs of a resumed run."""
        self._history = [
            {**asdict(metrics), "val_perplexity": metrics.val_perplexity} for metrics in history
        ]

    def _write_history(self) -> None:
        path = self.directory / HISTORY_FILE
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self._history, indent=2), encoding="utf-8")
        temporary.replace(path)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self._writer.close()

    def __enter__(self) -> RunTracker:
        """Return the tracker itself."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the tracker."""
        self.close()
