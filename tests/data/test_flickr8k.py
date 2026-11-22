from pathlib import Path

import pytest

from clipcap.data.flickr8k import (
    CAPTIONS_PER_IMAGE,
    CaptionFormatError,
    clean_caption,
    load_captions,
)

TOKEN_FILE = "".join(
    f"{image}#{index}\tA caption number {index} for {image} .\n"
    for image in ("first.jpg", "second.jpg")
    for index in range(CAPTIONS_PER_IMAGE)
)

CSV_FILE = "image,caption\n" + "".join(
    f"{image},A caption number {index} for {image} .\n"
    for image in ("first.jpg", "second.jpg")
    for index in range(CAPTIONS_PER_IMAGE)
)


def _write(tmp_path: Path, text: str, name: str = "captions.txt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A dog runs .", "A dog runs"),
        ("  A   dog\truns  ", "A dog runs"),
        ("A dog , a cat and a bird .", "A dog, a cat and a bird"),
        ("Is it a dog ?", "Is it a dog?"),
        ("A Black Dog", "A Black Dog"),
        (" . ", ""),
        ("", ""),
    ],
)
def test_clean_caption(raw: str, expected: str) -> None:
    assert clean_caption(raw) == expected


@pytest.mark.parametrize("content", [TOKEN_FILE, CSV_FILE], ids=["token", "csv"])
def test_load_captions_groups_five_captions_per_image(tmp_path: Path, content: str) -> None:
    captions = load_captions(_write(tmp_path, content))
    assert list(captions) == ["first.jpg", "second.jpg"]
    assert all(len(items) == CAPTIONS_PER_IMAGE for items in captions.values())
    assert captions["second.jpg"][3] == "A caption number 3 for second.jpg"


def test_csv_captions_may_contain_commas_and_quotes(tmp_path: Path) -> None:
    content = 'image,caption\na.jpg,A dog , a cat .\nb.jpg,"A ""big"" dog, running ."\n'
    captions = load_captions(_write(tmp_path, content))
    assert captions == {"a.jpg": ["A dog, a cat"], "b.jpg": ['A "big" dog, running']}


def test_blank_lines_and_empty_captions_are_skipped(tmp_path: Path) -> None:
    content = "a.jpg#0\tA dog .\n\na.jpg#1\t .\na.jpg#2\tA cat .\n"
    assert load_captions(_write(tmp_path, content)) == {"a.jpg": ["A dog", "A cat"]}


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="caption file not found"):
        load_captions(tmp_path / "absent.txt")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "is empty"),
        ("\n  \n", "is empty"),
        ("just some text\n", "unrecognised format"),
        ("a.jpg#0\tA dog .\nbroken line\n", r":2: expected"),
        ("image,caption\nonly-one-column\n", r":2: expected"),
        ("image,caption\n,missing image name\n", r":2: expected"),
        ("a.jpg#0\t .\n", "no usable caption"),
    ],
)
def test_malformed_files_are_rejected(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(CaptionFormatError, match=message):
        load_captions(_write(tmp_path, content))
