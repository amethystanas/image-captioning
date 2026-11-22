"""Download Flickr8k and check that a local copy is complete.

Expected layout under the dataset root, as produced by :func:`download_flickr8k`::

    Flicker8k_Dataset/*.jpg            (the directory name typo is upstream's)
    Flickr8k.token.txt
    Flickr_8k.trainImages.txt
    Flickr_8k.devImages.txt
    Flickr_8k.testImages.txt
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from clipcap.data.flickr8k import CAPTIONS_PER_IMAGE, load_captions
from clipcap.data.splits import Splits, load_split_files
from clipcap.utils import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k"
IMAGES_URL = f"{_BASE_URL}/Flickr8k_Dataset.zip"
TEXT_URL = f"{_BASE_URL}/Flickr8k_text.zip"


class DatasetError(RuntimeError):
    """Raised when the local dataset is missing, incomplete or corrupted."""


@dataclass(frozen=True)
class Flickr8kPaths:
    """Paths of the dataset files under a root directory."""

    root: Path

    @property
    def images_dir(self) -> Path:
        """Directory holding the JPEG images."""
        return self.root / "Flicker8k_Dataset"

    @property
    def captions(self) -> Path:
        """Caption file."""
        return self.root / "Flickr8k.token.txt"

    @property
    def train_split(self) -> Path:
        """Names of the training images."""
        return self.root / "Flickr_8k.trainImages.txt"

    @property
    def val_split(self) -> Path:
        """Names of the validation images."""
        return self.root / "Flickr_8k.devImages.txt"

    @property
    def test_split(self) -> Path:
        """Names of the test images."""
        return self.root / "Flickr_8k.testImages.txt"

    def image(self, name: str) -> Path:
        """Path of the image called ``name``."""
        return self.images_dir / name

    def required(self) -> tuple[Path, ...]:
        """Every path that must exist for the dataset to be usable."""
        return (
            self.images_dir,
            self.captions,
            self.train_split,
            self.val_split,
            self.test_split,
        )


@dataclass(frozen=True)
class Flickr8k:
    """A verified local copy of the dataset."""

    paths: Flickr8kPaths
    captions: dict[str, list[str]]
    splits: Splits


def load_flickr8k(root: str | Path) -> Flickr8k:
    """Load captions and splits from ``root`` after checking its integrity.

    Raises :class:`DatasetError` if files are missing, if a split references
    an image without captions or if an image file is absent or empty.
    """
    paths = Flickr8kPaths(Path(root))
    missing = [str(path) for path in paths.required() if not path.exists()]
    if missing:
        raise DatasetError(f"incomplete dataset under {paths.root}, missing: {missing}")

    captions = load_captions(paths.captions)
    splits = load_split_files(paths.train_split, paths.val_split, paths.test_split)
    for split, images in splits.items():
        problems = [_image_problem(paths, captions, image) for image in images]
        problems = [problem for problem in problems if problem]
        if problems:
            raise DatasetError(
                f"split '{split}' has {len(problems)} invalid image(s), e.g. {problems[0]}"
            )
    return Flickr8k(paths=paths, captions=captions, splits=splits)


def _image_problem(paths: Flickr8kPaths, captions: dict[str, list[str]], image: str) -> str:
    path = paths.image(image)
    if not path.is_file():
        return f"'{image}': file not found"
    if path.stat().st_size == 0:
        return f"'{image}': file is empty"
    count = len(captions.get(image, ()))
    if count != CAPTIONS_PER_IMAGE:
        return f"'{image}': {count} caption(s) instead of {CAPTIONS_PER_IMAGE}"
    return ""


def download_file(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` unless it is already there.

    The payload goes to a temporary file first, so an interrupted download
    never leaves a truncated file at ``destination``.
    """
    if destination.is_file():
        logger.info("already downloaded: %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    logger.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1 << 20)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a zip archive, refusing entries that would escape ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if not name.startswith("__MACOSX/")]
            for name in members:
                if not (root / name).resolve().is_relative_to(root):
                    raise DatasetError(f"{archive}: unsafe path in archive: '{name}'")
            bundle.extractall(destination, members=members)
    except zipfile.BadZipFile as exc:
        raise DatasetError(f"{archive} is not a valid zip archive") from exc


def download_flickr8k(
    root: str | Path,
    *,
    images_url: str = IMAGES_URL,
    text_url: str = TEXT_URL,
    keep_archives: bool = False,
) -> Flickr8k:
    """Download and extract Flickr8k into ``root``, then verify it.

    Nothing is downloaded when a complete copy is already present.
    """
    root = Path(root)
    paths = Flickr8kPaths(root)
    if all(path.exists() for path in paths.required()):
        logger.info("dataset already present under %s", root)
        return load_flickr8k(root)

    for url in (text_url, images_url):
        archive = download_file(url, root / Path(url).name)
        extract_archive(archive, root)
        if not keep_archives:
            archive.unlink()
    return load_flickr8k(root)
