"""Fixtures shared by the whole test suite."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from clipcap.data.flickr8k import CAPTIONS_PER_IMAGE
from clipcap.pipeline import Components

# The models under test are tiny: one thread is faster than paying for thread
# synchronisation, and it keeps timings stable on shared CI runners.
torch.set_num_threads(1)

SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "val": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}
TINY_SPLITS = {
    "train": ("train_0.jpg", "train_1.jpg", "train_2.jpg", "train_3.jpg"),
    "val": ("val_0.jpg", "val_1.jpg"),
    "test": ("test_0.jpg", "test_1.jpg"),
}
_SUBJECTS = ("dog", "cat", "child", "man", "woman", "bird", "horse", "boy")
_COLOURS = ((200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30))


def build_tiny_flickr8k(root: Path) -> Path:
    """Write a miniature but structurally faithful Flickr8k under ``root``."""
    images_dir = root / "Flicker8k_Dataset"
    images_dir.mkdir(parents=True)
    lines = []
    names = [name for split in TINY_SPLITS.values() for name in split]
    for position, name in enumerate(names):
        colour = _COLOURS[position % len(_COLOURS)]
        Image.new("RGB", (32, 24), colour).save(images_dir / name)
        subject = _SUBJECTS[position % len(_SUBJECTS)]
        lines += [
            f"{name}#{index}\tA {subject} is playing outside , take {index} ."
            for index in range(CAPTIONS_PER_IMAGE)
        ]
    (root / "Flickr8k.token.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for split, filename in SPLIT_FILES.items():
        (root / filename).write_text("\n".join(TINY_SPLITS[split]) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def tiny_flickr8k(tmp_path: Path) -> Path:
    """Root directory of a miniature Flickr8k (4 train, 2 val, 2 test images)."""
    return build_tiny_flickr8k(tmp_path / "flickr8k")


@pytest.fixture
def components() -> Components:
    """Offline stand-ins for CLIP, the GPT-2 tokenizer and GPT-2."""
    from tests.helpers import make_offline_components

    return make_offline_components()


def write_experiment_config(path: Path, dataset_root: Path, **sections: dict[str, object]) -> Path:
    """Write a small CPU experiment under ``path.parent`` and return ``path``.

    ``sections`` overrides keys of the ``model``, ``training`` and
    ``generation`` sections.
    """
    import yaml

    config: dict[str, dict[str, object]] = {
        "data": {
            "root": str(dataset_root),
            "features_dir": str(path.parent / "features"),
            "num_workers": 0,
        },
        "model": {"prefix_length": 3, "freeze_gpt2": False},
        "training": {
            "output_dir": str(path.parent / "run"),
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 5e-3,
            "warmup_steps": 1,
            "device": "cpu",
        },
        "generation": {"strategy": "greedy", "max_new_tokens": 10},
    }
    for section, overrides in sections.items():
        config[section].update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tiny_flickr8k: Path, tmp_path: Path) -> Path:
    """A two-epoch CPU experiment on the miniature dataset."""
    return write_experiment_config(tmp_path / "experiment.yaml", tiny_flickr8k)
