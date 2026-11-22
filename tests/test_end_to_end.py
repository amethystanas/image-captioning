"""The whole pipeline, chained through the command line."""

import json
from pathlib import Path

import pytest
import torch

from clipcap.cli import main
from clipcap.pipeline import Components
from clipcap.training.trainer import read_checkpoint
from tests.conftest import write_experiment_config
from tests.helpers import make_offline_components


def _run(components: Components | None, *arguments: object) -> int:
    return main([str(argument) for argument in arguments], components)


def _pipeline(config: Path, components: Components) -> dict[str, object]:
    """Run extract, train and evaluate, and return the artefacts worth comparing."""
    for command in ("extract", "train", "evaluate"):
        assert _run(components, command, "--config", config) == 0, command
    run = config.parent / "run"
    return {
        "history": json.loads((run / "history.json").read_text(encoding="utf-8")),
        "report": json.loads((run / "eval_test_beam.json").read_text(encoding="utf-8")),
        "weights": read_checkpoint(run / "best.pt")["model"],
    }


@pytest.mark.parametrize(("mapper_type", "freeze_gpt2"), [("mlp", False), ("transformer", True)])
def test_full_pipeline_is_consistent_and_reproducible(
    tiny_flickr8k: Path, tmp_path: Path, mapper_type: str, freeze_gpt2: bool
) -> None:
    sections = {
        "model": {
            "mapper_type": mapper_type,
            "freeze_gpt2": freeze_gpt2,
            "mapper_num_layers": 1,
            "mapper_num_heads": 2,
        },
        "training": {"epochs": 3},
        "generation": {"strategy": "beam", "beam_size": 2},
    }
    first_config = write_experiment_config(tmp_path / "a" / "c.yaml", tiny_flickr8k, **sections)
    second_config = write_experiment_config(tmp_path / "b" / "c.yaml", tiny_flickr8k, **sections)
    first = _pipeline(first_config, make_offline_components())
    second = _pipeline(second_config, make_offline_components())

    history, report = first["history"], first["report"]
    assert isinstance(history, list)
    assert isinstance(report, dict)
    assert [item["epoch"] for item in history] == [1, 2, 3]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert report["num_images"] == 2
    assert report["generation"]["strategy"] == "beam"
    assert all(len(item["references"]) == 5 for item in report["captions"])
    assert all(0.0 <= value <= 10.0 for value in report["scores"].values())

    best_epoch = min(history, key=lambda item: item["val_loss"])["epoch"]
    checkpoint = read_checkpoint(first_config.parent / "run" / "best.pt")
    assert checkpoint["state"]["epoch"] == best_epoch
    assert any(key.startswith("gpt2.") for key in checkpoint["model"]) is not freeze_gpt2

    # Same configuration and seed, different directories: identical results.
    assert second["history"] == history
    assert second["report"] == report
    weights, other_weights = first["weights"], second["weights"]
    assert isinstance(weights, dict)
    assert isinstance(other_weights, dict)
    assert all(torch.equal(value, other_weights[name]) for name, value in weights.items())


def test_interrupted_training_resumed_from_the_cli_matches_a_straight_run(
    tiny_flickr8k: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clipcap.training.trainer import Trainer

    sections = {"training": {"epochs": 4}}
    straight = write_experiment_config(tmp_path / "straight" / "c.yaml", tiny_flickr8k, **sections)
    resumed = write_experiment_config(tmp_path / "resumed" / "c.yaml", tiny_flickr8k, **sections)

    for config in (straight, resumed):
        assert _run(make_offline_components(), "extract", "--config", config) == 0
    assert _run(make_offline_components(), "train", "--config", straight) == 0

    original = Trainer.evaluate

    def crash_in_third_epoch(self: Trainer) -> float:
        if self.state.epoch == 2:
            raise KeyboardInterrupt
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "evaluate", crash_in_third_epoch)
        with pytest.raises(KeyboardInterrupt):
            _run(make_offline_components(), "train", "--config", resumed)
    assert _run(make_offline_components(), "train", "--config", resumed, "--resume") == 0

    def artefacts(config: Path) -> tuple[object, dict[str, torch.Tensor]]:
        run = config.parent / "run"
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        return history, read_checkpoint(run / "last.pt")["model"]

    straight_history, straight_weights = artefacts(straight)
    resumed_history, resumed_weights = artefacts(resumed)
    assert resumed_history == straight_history
    assert all(
        torch.equal(value, resumed_weights[name]) for name, value in straight_weights.items()
    )


@pytest.mark.slow
def test_real_clip_and_gpt2_pipeline(
    tiny_flickr8k: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One epoch with the pretrained CLIP ViT-B/32 and GPT-2 on the miniature dataset."""
    config = write_experiment_config(
        tmp_path / "real" / "c.yaml",
        tiny_flickr8k,
        model={"prefix_length": 4, "freeze_gpt2": True},
        training={"epochs": 1, "batch_size": 4, "learning_rate": 1e-3},
        generation={"strategy": "beam", "beam_size": 2, "max_new_tokens": 8},
    )
    for command in ("extract", "train", "evaluate"):
        assert _run(None, command, "--config", config) == 0, command
    run = config.parent / "run"
    report = json.loads((run / "eval_test_beam.json").read_text(encoding="utf-8"))
    assert report["num_images"] == 2
    assert all(isinstance(item["caption"], str) for item in report["captions"])
    # A frozen GPT-2 (about 500 MB) is not stored: the checkpoint only holds the
    # mapper and its optimiser state.
    assert (run / "best.pt").stat().st_size < 150 * 1024 * 1024

    capsys.readouterr()
    image = tiny_flickr8k / "Flicker8k_Dataset" / "test_0.jpg"
    assert _run(None, "caption", "--config", config, image) == 0
    path, caption = capsys.readouterr().out.strip().split("\t")
    assert path == str(image)
    assert caption == report["captions"][0]["caption"]
