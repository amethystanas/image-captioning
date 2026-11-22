import itertools
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from clipcap.config import TrainingConfig
from clipcap.data.dataset import CaptionBatch, CaptionDataset
from clipcap.models.clipcap import ClipCaptionModel
from clipcap.training.trainer import (
    CheckpointError,
    EpochMetrics,
    Trainer,
    TrainingState,
    linear_warmup_decay,
    load_model_weights,
    read_checkpoint,
)
from tests.helpers import WordTokenizer, make_features, make_tiny_model


@pytest.fixture(autouse=True)
def _isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoints go to a relative ``outputs/`` directory: keep it out of the repository."""
    monkeypatch.chdir(tmp_path)


SUBJECTS = ("dog", "cat", "boy", "girl", "man", "bird", "horse", "child")


def make_loaders(batch_size: int = 4) -> tuple[DataLoader[CaptionBatch], DataLoader[CaptionBatch]]:
    """Eight training images and four validation images, two captions each."""
    names = tuple(f"{index}.jpg" for index in range(12))
    features = make_features(names)
    captions = {
        name: [
            f"a {SUBJECTS[index % 8]} runs in the park",
            f"the {SUBJECTS[index % 8]} is outside",
        ]
        for index, name in enumerate(names)
    }
    tokenizer = WordTokenizer()

    def loader(subset: tuple[str, ...], shuffle: bool) -> DataLoader[CaptionBatch]:
        dataset = CaptionDataset(features, captions, subset, tokenizer, eos_token_id=0)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate
        )

    return loader(names[:8], True), loader(names[8:], False)


def make_trainer(*, model: ClipCaptionModel | None = None, **overrides: object) -> Trainer:
    settings: dict[str, object] = {
        "epochs": 3,
        "learning_rate": 5e-3,
        "warmup_steps": 2,
        "device": "cpu",
        "patience": 0,
        **overrides,
    }
    config = TrainingConfig(**settings)  # type: ignore[arg-type]
    train_loader, val_loader = make_loaders()
    return Trainer(model or make_tiny_model(freeze_gpt2=False), train_loader, val_loader, config)


def test_training_reduces_train_and_validation_loss() -> None:
    trainer = make_trainer(epochs=4)
    initial = trainer.evaluate()
    state = trainer.fit()

    assert state.epoch == 4
    assert state.step == 4 * len(trainer.train_loader)
    assert [metrics.epoch for metrics in state.history] == [1, 2, 3, 4]
    assert state.history[-1].train_loss < state.history[0].train_loss
    assert state.history[-1].val_loss < initial
    assert all(math.isfinite(metrics.val_loss) for metrics in state.history)


def test_best_epoch_tracks_the_lowest_validation_loss() -> None:
    state = make_trainer(epochs=3).fit()
    best = min(state.history, key=lambda metrics: metrics.val_loss)
    assert state.best_epoch == best.epoch
    assert state.best_val_loss == best.val_loss


def test_training_is_deterministic_for_a_given_seed() -> None:
    first = make_trainer(seed=5).fit()
    second = make_trainer(seed=5).fit()
    other = make_trainer(seed=6).fit()
    assert first.history == second.history
    assert first.history != other.history


def test_only_the_mapper_is_updated_when_gpt2_is_frozen() -> None:
    model = make_tiny_model(freeze_gpt2=True)
    gpt2_before = {name: value.clone() for name, value in model.gpt2.state_dict().items()}
    mapper_before = {name: value.clone() for name, value in model.mapper.state_dict().items()}
    make_trainer(model=model, epochs=1).fit()

    for name, value in model.gpt2.state_dict().items():
        assert torch.equal(value, gpt2_before[name]), name
    assert any(
        not torch.equal(value, mapper_before[name])
        for name, value in model.mapper.state_dict().items()
    )


def test_evaluate_runs_in_eval_mode_without_touching_the_weights() -> None:
    trainer = make_trainer()
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    trainer.model.train()
    first = trainer.evaluate()
    assert not trainer.model.training
    assert trainer.evaluate() == first
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_validation_loss_is_a_per_token_mean_independent_of_batching() -> None:
    model = make_tiny_model(freeze_gpt2=False)
    config = TrainingConfig(device="cpu")
    losses = []
    for batch_size in (1, 3, 8):
        train_loader, val_loader = make_loaders(batch_size)
        losses.append(Trainer(model, train_loader, val_loader, config).evaluate())
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
    assert losses[0] == pytest.approx(losses[2], rel=1e-5)


def test_learning_rate_warms_up_then_decays_to_zero() -> None:
    trainer = make_trainer(epochs=2, warmup_steps=3, learning_rate=1e-2)
    rates = []
    for _ in range(trainer.total_steps):
        rates.append(trainer.scheduler.get_last_lr()[0])
        trainer.optimizer.step()
        trainer.scheduler.step()
    assert rates[:4] == pytest.approx([0.0025, 0.005, 0.0075, 0.01])
    assert all(later < earlier for earlier, later in itertools.pairwise(rates[3:]))
    assert trainer.scheduler.get_last_lr()[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.2), (3, 0.8), (4, 1.0), (9, 0.5), (14, 0.0), (20, 0.0)],
)
def test_linear_warmup_decay_factor(step: int, expected: float) -> None:
    assert linear_warmup_decay(step, warmup_steps=4, total_steps=14) == pytest.approx(expected)


def test_no_warmup_starts_at_full_learning_rate() -> None:
    assert linear_warmup_decay(0, warmup_steps=0, total_steps=10) == pytest.approx(1.0)


def test_gradients_are_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(max_norm)
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    trainer = make_trainer(epochs=1, max_grad_norm=0.25)
    trainer.fit()
    assert calls == [0.25] * len(trainer.train_loader)


def test_early_stopping_after_patience_epochs_without_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(epochs=10, patience=2)
    losses = iter([3.0, 2.0, 2.5, 2.1, 1.0, 1.0])
    monkeypatch.setattr(trainer, "evaluate", lambda: next(losses))
    state = trainer.fit()
    assert state.epoch == 4
    assert state.best_epoch == 2
    assert state.best_val_loss == 2.0
    assert state.epochs_without_improvement == 2


def test_patience_zero_disables_early_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = make_trainer(epochs=4, patience=0)
    monkeypatch.setattr(trainer, "evaluate", lambda: 1.0)
    assert trainer.fit().epoch == 4


def test_non_finite_loss_aborts_training() -> None:
    trainer = make_trainer(epochs=1)
    with torch.no_grad():
        next(iter(trainer.model.mapper.parameters())).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite training loss at step 0"):
        trainer.fit()


def test_mixed_precision_is_only_enabled_on_cuda() -> None:
    assert make_trainer(mixed_precision=True).use_amp is False


def test_empty_loaders_and_frozen_models_are_rejected() -> None:
    train_loader, val_loader = make_loaders()
    config = TrainingConfig(device="cpu")
    with pytest.raises(ValueError, match="must not be empty"):
        Trainer(make_tiny_model(), [], val_loader, config)  # type: ignore[arg-type]
    model = make_tiny_model()
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameter"):
        Trainer(model, train_loader, val_loader, config)


def test_training_state_round_trip() -> None:
    state = TrainingState(
        epoch=2,
        step=8,
        best_val_loss=1.5,
        best_epoch=1,
        epochs_without_improvement=1,
        history=[EpochMetrics(1, 2.0, 1.5, 1e-3), EpochMetrics(2, 1.8, 1.6, 5e-4)],
    )
    assert TrainingState.from_dict(state.to_dict()) == state
    assert state.history[0].val_perplexity == pytest.approx(math.exp(1.5))


def _weights(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in trainer.model.state_dict().items()}


def _assert_same_weights(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> None:
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert torch.equal(value, second[name]), name


def test_checkpoints_are_written_every_epoch_and_best_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = make_trainer(epochs=3)
    losses = iter([2.0, 1.0, 1.5])
    monkeypatch.setattr(trainer, "evaluate", lambda: next(losses))
    trainer.fit()

    assert sorted(path.name for path in trainer.output_dir.iterdir()) == ["best.pt", "last.pt"]
    assert read_checkpoint(trainer.last_checkpoint)["state"]["epoch"] == 3
    assert read_checkpoint(trainer.best_checkpoint)["state"]["epoch"] == 2


@pytest.mark.parametrize("freeze_gpt2", [True, False])
def test_interrupted_run_resumes_to_the_same_result(
    monkeypatch: pytest.MonkeyPatch, freeze_gpt2: bool
) -> None:
    def trainer_for(directory: str) -> Trainer:
        return make_trainer(
            model=make_tiny_model(freeze_gpt2=freeze_gpt2), epochs=4, output_dir=directory
        )

    reference = trainer_for("reference")
    reference.fit()

    interrupted = trainer_for("resumed")
    original_evaluate = interrupted.evaluate

    def evaluate_then_crash() -> float:
        if interrupted.state.epoch == 2:
            raise KeyboardInterrupt
        return original_evaluate()

    monkeypatch.setattr(interrupted, "evaluate", evaluate_then_crash)
    with pytest.raises(KeyboardInterrupt):
        interrupted.fit()
    assert interrupted.state.epoch == 2

    resumed = trainer_for("resumed")
    assert resumed.resume() is True
    assert resumed.state.epoch == 2
    resumed.fit()

    assert resumed.state == reference.state
    _assert_same_weights(_weights(resumed), _weights(reference))
    assert resumed.optimizer.state_dict()["state"].keys() == (
        reference.optimizer.state_dict()["state"].keys()
    )
    assert resumed.scheduler.get_last_lr() == reference.scheduler.get_last_lr()


def test_resume_without_checkpoint_starts_from_scratch() -> None:
    trainer = make_trainer()
    assert trainer.resume() is False
    assert trainer.state.epoch == 0


def test_resuming_a_finished_run_does_nothing() -> None:
    make_trainer(epochs=2).fit()
    trainer = make_trainer(epochs=2)
    trainer.resume()
    before = _weights(trainer)
    assert trainer.fit().epoch == 2
    _assert_same_weights(_weights(trainer), before)


def test_resuming_an_early_stopped_run_does_not_train_further(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_trainer(epochs=10, patience=1)
    losses = iter([1.0, 2.0])
    monkeypatch.setattr(first, "evaluate", lambda: next(losses))
    assert first.fit().epoch == 2

    second = make_trainer(epochs=10, patience=1)
    second.resume()
    assert second.fit().epoch == 2


def test_frozen_gpt2_is_left_out_of_checkpoints() -> None:
    frozen = make_trainer(model=make_tiny_model(freeze_gpt2=True), epochs=1, output_dir="frozen")
    tuned = make_trainer(model=make_tiny_model(freeze_gpt2=False), epochs=1, output_dir="tuned")
    frozen.fit()
    tuned.fit()
    frozen_keys = read_checkpoint(frozen.last_checkpoint)["model"].keys()
    tuned_keys = read_checkpoint(tuned.last_checkpoint)["model"].keys()
    assert frozen_keys
    assert all(key.startswith("mapper.") for key in frozen_keys)
    assert any(key.startswith("gpt2.") for key in tuned_keys)
    assert frozen.last_checkpoint.stat().st_size < tuned.last_checkpoint.stat().st_size


def test_load_model_weights_restores_the_best_model() -> None:
    trainer = make_trainer(model=make_tiny_model(freeze_gpt2=True), epochs=2)
    trainer.fit()
    fresh = make_tiny_model(freeze_gpt2=True, seed=0)
    state = load_model_weights(trainer.best_checkpoint, fresh)
    assert state.epoch == trainer.state.best_epoch
    assert state.best_val_loss == trainer.state.best_val_loss
    if trainer.state.best_epoch == trainer.state.epoch:
        _assert_same_weights(fresh.state_dict(), trainer.model.state_dict())


def test_checkpoint_of_another_architecture_is_refused() -> None:
    trainer = make_trainer(model=make_tiny_model(freeze_gpt2=True), epochs=1)
    trainer.fit()
    with pytest.raises(CheckpointError, match="does not match the model"):
        load_model_weights(trainer.last_checkpoint, make_tiny_model(freeze_gpt2=False))
    with pytest.raises(CheckpointError, match="does not match the model"):
        load_model_weights(trainer.last_checkpoint, make_tiny_model(prefix_length=5))


def test_resuming_with_another_architecture_is_refused() -> None:
    make_trainer(model=make_tiny_model(prefix_length=3), epochs=1).fit()
    other = make_trainer(model=make_tiny_model(prefix_length=5), epochs=1)
    with pytest.raises(CheckpointError, match="tensor\\(s\\) of a different shape"):
        other.resume()


def test_resuming_with_a_different_schedule_is_refused() -> None:
    make_trainer(epochs=2).fit()
    with pytest.raises(CheckpointError, match="epochs and batch size must match"):
        make_trainer(epochs=5).resume()


def test_unreadable_checkpoints_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        read_checkpoint(tmp_path / "absent.pt")
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"garbage")
    with pytest.raises(CheckpointError, match="not a valid checkpoint"):
        read_checkpoint(broken)
    torch.save({"version": 99}, broken)
    with pytest.raises(CheckpointError, match="unsupported checkpoint version 99"):
        read_checkpoint(broken)
