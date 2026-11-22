"""Train / validation / test splits of Flickr8k.

The reference split is Karpathy's (6000 / 1000 / 1000 images), distributed as
``dataset_flickr8k.json``. It matches the official split files shipped with
the original dataset (``Flickr_8k.{train,dev,test}Images.txt``), which are
supported too. A seeded random split is available as a fallback when neither
is at hand.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

SPLIT_NAMES = ("train", "val", "test")
KARPATHY_SIZES = {"train": 6000, "val": 1000, "test": 1000}

# Karpathy's files label extra training images "restval"; Flickr8k has none,
# but folding them into train keeps the loader valid for Flickr30k / COCO.
_KARPATHY_ALIASES = {"train": "train", "restval": "train", "val": "val", "test": "test"}


class SplitError(ValueError):
    """Raised when split definitions are malformed or inconsistent."""


@dataclass(frozen=True)
class Splits:
    """Image names of each split, in a stable order."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check that no split is empty and that splits do not overlap."""
        seen: dict[str, str] = {}
        for name, images in self.items():
            if not images:
                raise SplitError(f"split '{name}' is empty")
            if len(set(images)) != len(images):
                raise SplitError(f"split '{name}' contains duplicate images")
            for image in images:
                if image in seen:
                    raise SplitError(f"image '{image}' is in both '{seen[image]}' and '{name}'")
                seen[image] = name

    def __getitem__(self, name: str) -> tuple[str, ...]:
        """Return the images of split ``name``."""
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split '{name}', expected one of {SPLIT_NAMES}")
        images: tuple[str, ...] = getattr(self, name)
        return images

    def items(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        """Iterate over ``(split name, images)`` pairs."""
        for name in SPLIT_NAMES:
            yield name, self[name]

    def sizes(self) -> dict[str, int]:
        """Return the number of images in each split."""
        return {name: len(images) for name, images in self.items()}

    def check_against(self, available: Iterable[str]) -> None:
        """Raise if a split references an image absent from ``available``."""
        known = set(available)
        for name, images in self.items():
            missing = [image for image in images if image not in known]
            if missing:
                raise SplitError(
                    f"split '{name}' references {len(missing)} unknown image(s), "
                    f"e.g. '{missing[0]}'"
                )


def load_karpathy_splits(path: str | Path) -> Splits:
    """Read the splits from Karpathy's ``dataset_flickr8k.json``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Karpathy split file not found: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        entries = content["images"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SplitError(f"{path} is not a Karpathy split file: {exc!r}") from exc

    images: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for position, entry in enumerate(entries):
        try:
            filename, raw_split = entry["filename"], entry["split"]
        except (KeyError, TypeError) as exc:
            raise SplitError(f"{path}: entry {position} lacks 'filename' or 'split'") from exc
        if raw_split not in _KARPATHY_ALIASES:
            raise SplitError(f"{path}: entry {position} has unknown split '{raw_split}'")
        images[_KARPATHY_ALIASES[raw_split]].append(filename)
    return Splits(**{name: tuple(items) for name, items in images.items()})


def load_split_files(train: str | Path, val: str | Path, test: str | Path) -> Splits:
    """Read the splits from the official one-image-name-per-line files."""
    return Splits(train=_read_names(train), val=_read_names(val), test=_read_names(test))


def _read_names(path: str | Path) -> tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(line.strip() for line in lines if line.strip())


def make_random_splits(
    images: Iterable[str], *, val_size: int, test_size: int, seed: int = 42
) -> Splits:
    """Split ``images`` at random, deterministically for a given ``seed``.

    The result does not depend on the order in which ``images`` is given.
    """
    if val_size <= 0 or test_size <= 0:
        raise SplitError("val_size and test_size must be positive")
    pool = sorted(set(images))
    if len(pool) <= val_size + test_size:
        raise SplitError(
            f"{len(pool)} image(s) cannot provide {val_size} validation and "
            f"{test_size} test images plus a training set"
        )
    random.Random(seed).shuffle(pool)
    return Splits(
        test=tuple(pool[:test_size]),
        val=tuple(pool[test_size : test_size + val_size]),
        train=tuple(pool[test_size + val_size :]),
    )
