"""Fixtures shared by the whole test suite."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from clipcap.data.flickr8k import CAPTIONS_PER_IMAGE

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
