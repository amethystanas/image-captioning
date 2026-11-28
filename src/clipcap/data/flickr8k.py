r"""Flickr8k caption files: parsing and text normalisation.

Two distributions of the annotations are supported:

* the original ``Flickr8k.token.txt`` (``<image>#<index>\t<caption>`` per line);
* the Kaggle ``captions.txt`` (CSV with an ``image,caption`` header).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

CAPTIONS_PER_IMAGE = 5

_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,!?;:])")
_TOKEN_LINE = re.compile(r"^(?P<image>.+?)#\d+\t(?P<caption>.*)$")


class CaptionFormatError(ValueError):
    """Raised when a caption file does not follow a supported format."""


def clean_caption(text: str) -> str:
    """Normalise a raw caption.

    Whitespace is collapsed, the space the dataset puts before punctuation is
    removed and the trailing full stop is dropped. Casing is preserved because
    GPT-2 is case-sensitive.
    """
    text = _WHITESPACE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return text.rstrip(" .").strip()


def load_captions(path: str | Path) -> dict[str, list[str]]:
    """Read a caption file and return cleaned captions grouped by image name.

    Image order and caption order follow the file. Captions that are empty
    after cleaning are dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"caption file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty = [(number, line) for number, line in enumerate(lines, start=1) if line.strip()]
    if not non_empty:
        raise CaptionFormatError(f"{path} is empty")

    first_line = non_empty[0][1]
    if _TOKEN_LINE.match(first_line):
        pairs = _parse_token_lines(path, non_empty)
    elif first_line.strip().lower().replace(" ", "") == "image,caption":
        pairs = _parse_csv_lines(path, non_empty[1:])
    else:
        raise CaptionFormatError(
            f"{path}: unrecognised format, expected '<image>#<n>\\t<caption>' lines "
            "or a CSV with an 'image,caption' header"
        )

    captions: dict[str, list[str]] = {}
    for image, raw_caption in pairs:
        cleaned = clean_caption(raw_caption)
        if cleaned:
            captions.setdefault(image, []).append(cleaned)
    if not captions:
        raise CaptionFormatError(f"{path} contains no usable caption")
    return captions


def _parse_token_lines(path: Path, lines: list[tuple[int, str]]) -> list[tuple[str, str]]:
    pairs = []
    for number, line in lines:
        match = _TOKEN_LINE.match(line)
        if match is None:
            raise CaptionFormatError(f"{path}:{number}: expected '<image>#<n>\\t<caption>'")
        pairs.append((match["image"].strip(), match["caption"]))
    return pairs


def _parse_csv_lines(path: Path, lines: list[tuple[int, str]]) -> list[tuple[str, str]]:
    pairs = []
    for number, line in lines:
        row = next(csv.reader([line]))
        if len(row) < 2 or not row[0].strip():
            raise CaptionFormatError(f"{path}:{number}: expected '<image>,<caption>'")
        # Unquoted captions may themselves contain commas.
        pairs.append((row[0].strip(), ",".join(row[1:])))
    return pairs
