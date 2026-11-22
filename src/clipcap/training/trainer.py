"""Training loop of the captioning model."""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterable, Sized
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.optim.lr_scheduler import LambdaLR

from clipcap.config import TrainingConfig
from clipcap.data.dataset import CaptionBatch
from clipcap.models.clipcap import ClipCaptionModel
from clipcap.utils import get_logger, resolve_device, seed_everything

logger = get_logger(__name__)

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"
_CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is unreadable or does not fit the run."""


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint file written by :meth:`Trainer.save_checkpoint`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        version = checkpoint["version"]
    except Exception as exc:
        raise CheckpointError(f"{path} is not a valid checkpoint: {exc!r}") from exc
    if version != _CHECKPOINT_VERSION:
        raise CheckpointError(f"{path}: unsupported checkpoint version {version}")
    return checkpoint


def load_model_weights(path: str | Path, model: ClipCaptionModel) -> TrainingState:
    """Load only the model weights of a checkpoint, for evaluation or inference.

    Returns the training state stored with them (epoch, validation loss, history).
    """
    checkpoint = read_checkpoint(path)
    try:
        model.load_checkpoint_state(checkpoint["model"])
    except ValueError as exc:
        raise CheckpointError(f"{path}: {exc}") from exc
    return TrainingState.from_dict(checkpoint["state"])


class BatchLoader(Iterable[CaptionBatch], Sized, Protocol):
    """A re-iterable, sized source of batches, such as a ``DataLoader``."""


@dataclass(frozen=True)
class EpochMetrics:
    """Losses of one epoch; both are mean negative log-likelihoods per token."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float

    @property
    def val_perplexity(self) -> float:
        """Perplexity of the model on the validation captions."""
        return math.exp(self.val_loss)


@dataclass
class TrainingState:
    """Everything that evolves during training besides the weights."""

    epoch: int = 0
    step: int = 0
    best_val_loss: float = math.inf
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    history: list[EpochMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingState:
        """Rebuild a state written by :meth:`to_dict`."""
        history = [EpochMetrics(**item) for item in raw["history"]]
        return cls(**{**raw, "history": history})


def linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate factor: linear warm-up to 1, then linear decay to 0."""
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    remaining = total_steps - step
    return max(0.0, remaining / max(1, total_steps - warmup_steps))


class Trainer:
    """Optimise a :class:`ClipCaptionModel` and track its validation loss."""

    def __init__(
        self,
        model: ClipCaptionModel,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        config: TrainingConfig,
    ) -> None:
        """Prepare the optimiser and the learning-rate schedule.

        The schedule spans ``config.epochs`` epochs of ``len(train_loader)``
        steps. Only parameters that require gradients are optimised.
        """
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("training and validation loaders must not be empty")
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.state = TrainingState()

        parameters = model.trainable_parameters()
        if not parameters:
            raise ValueError("the model has no trainable parameter")
        self.optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.total_steps = config.epochs * len(train_loader)
        self.scheduler = LambdaLR(
            self.optimizer,
            lambda step: linear_warmup_decay(
                step, warmup_steps=config.warmup_steps, total_steps=self.total_steps
            ),
        )
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.output_dir = Path(config.output_dir)

    @property
    def last_checkpoint(self) -> Path:
        """Checkpoint written at the end of every epoch."""
        return self.output_dir / LAST_CHECKPOINT

    @property
    def best_checkpoint(self) -> Path:
        """Copy of the checkpoint with the lowest validation loss."""
        return self.output_dir / BEST_CHECKPOINT

    def save_checkpoint(self, path: Path) -> None:
        """Write everything needed to resume training, atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": _CHECKPOINT_VERSION,
            "model": self.model.checkpoint_state(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": self.state.to_dict(),
            "total_steps": self.total_steps,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: Path) -> None:
        """Restore weights, optimiser, schedule and progress from ``path``."""
        checkpoint = read_checkpoint(path)
        if checkpoint["total_steps"] != self.total_steps:
            raise CheckpointError(
                f"{path} comes from a run of {checkpoint['total_steps']} steps, this one has "
                f"{self.total_steps}: epochs and batch size must match to resume"
            )
        try:
            self.model.load_checkpoint_state(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            raise CheckpointError(f"{path}: {exc}") from exc
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.state = TrainingState.from_dict(checkpoint["state"])

    def resume(self) -> bool:
        """Continue from the last checkpoint of the output directory, if any.

        Returns whether a checkpoint was found and loaded.
        """
        if not self.last_checkpoint.is_file():
            return False
        self.load_checkpoint(self.last_checkpoint)
        logger.info("resumed from %s at epoch %d", self.last_checkpoint, self.state.epoch)
        return True

    def fit(self) -> TrainingState:
        """Train until ``config.epochs`` is reached or validation stops improving.

        A checkpoint is written after every epoch, and copied to
        :attr:`best_checkpoint` whenever the validation loss improves.
        """
        while self.state.epoch < self.config.epochs and not self.should_stop():
            epoch = self.state.epoch + 1
            # Reseeding per epoch makes the data order a function of (seed, epoch)
            # alone, so a resumed run sees the batches an uninterrupted one would.
            seed_everything(self.config.seed + epoch)
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            learning_rate = float(self.scheduler.get_last_lr()[0])
            metrics = EpochMetrics(epoch, train_loss, val_loss, learning_rate)
            self._record(metrics)
            logger.info(
                "epoch %d/%d | train %.4f | val %.4f | lr %.2e",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                metrics.learning_rate,
            )
            if self.should_stop():
                logger.info("early stopping: no improvement for %d epoch(s)", self.config.patience)
                break
        return self.state

    def should_stop(self) -> bool:
        """Tell whether patience is exhausted (a patience of 0 disables early stopping)."""
        patience = self.config.patience
        return patience > 0 and self.state.epochs_without_improvement >= patience

    def train_epoch(self) -> float:
        """Run one pass over the training data and return the mean token loss."""
        self.model.train()
        total_loss, total_tokens = 0.0, 0
        for batch in self.train_loader:
            loss, tokens = self._loss(batch)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"non-finite training loss at step {self.state.step}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.state.step += 1
            self._on_step(loss.item())
            total_loss += loss.item() * tokens
            total_tokens += tokens
        return total_loss / total_tokens

    @torch.no_grad()
    def evaluate(self) -> float:
        """Return the mean token loss on the validation data."""
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self.val_loader:
            loss, tokens = self._loss(batch)
            total_loss += loss.item() * tokens
            total_tokens += tokens
        return total_loss / total_tokens

    def _loss(self, batch: CaptionBatch) -> tuple[torch.Tensor, int]:
        batch = batch.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            output = self.model(batch.embeddings, batch.tokens, batch.mask)
        return output.loss, int(batch.mask.sum().item())

    def _record(self, metrics: EpochMetrics) -> None:
        state = self.state
        state.epoch = metrics.epoch
        state.history.append(metrics)
        if metrics.val_loss < state.best_val_loss:
            state.best_val_loss = metrics.val_loss
            state.best_epoch = metrics.epoch
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1
        self.save_checkpoint(self.last_checkpoint)
        if state.best_epoch == metrics.epoch:
            shutil.copyfile(self.last_checkpoint, self.best_checkpoint)
        self._on_epoch(metrics)

    def _on_step(self, loss: float) -> None:
        """Extension point called after every optimisation step."""

    def _on_epoch(self, metrics: EpochMetrics) -> None:
        """Extension point called after every epoch, once the state is updated."""
