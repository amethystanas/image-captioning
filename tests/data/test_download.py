import zipfile
from pathlib import Path

import pytest

from clipcap.data.download import (
    DatasetError,
    Flickr8kPaths,
    download_file,
    download_flickr8k,
    extract_archive,
    load_flickr8k,
)
from tests.conftest import SPLIT_FILES, TINY_SPLITS, build_tiny_flickr8k


def _zip(source: Path, archive: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in names:
            bundle.write(source / name, arcname=name)
    return archive


def _archives(tmp_path: Path) -> tuple[str, str]:
    """Build local images/text archives and return their ``file://`` URLs."""
    source = build_tiny_flickr8k(tmp_path / "source")
    images = [f"Flicker8k_Dataset/{name}" for split in TINY_SPLITS.values() for name in split]
    images_zip = _zip(source, tmp_path / "images.zip", images)
    text_zip = _zip(source, tmp_path / "text.zip", ["Flickr8k.token.txt", *SPLIT_FILES.values()])
    return images_zip.as_uri(), text_zip.as_uri()


def test_load_flickr8k_reads_captions_and_splits(tiny_flickr8k: Path) -> None:
    dataset = load_flickr8k(tiny_flickr8k)
    assert dataset.splits.sizes() == {"train": 4, "val": 2, "test": 2}
    assert len(dataset.captions) == 8
    assert dataset.captions["train_0.jpg"][0] == "A dog is playing outside, take 0"
    assert dataset.paths.image("val_0.jpg").is_file()


def test_missing_files_are_reported(tiny_flickr8k: Path) -> None:
    Flickr8kPaths(tiny_flickr8k).val_split.unlink()
    with pytest.raises(DatasetError, match=r"missing: .*devImages"):
        load_flickr8k(tiny_flickr8k)


def test_missing_root_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="incomplete dataset"):
        load_flickr8k(tmp_path / "nowhere")


def test_missing_image_is_reported(tiny_flickr8k: Path) -> None:
    Flickr8kPaths(tiny_flickr8k).image("test_1.jpg").unlink()
    with pytest.raises(DatasetError, match=r"'test' has 1 invalid.*file not found"):
        load_flickr8k(tiny_flickr8k)


def test_empty_image_is_reported(tiny_flickr8k: Path) -> None:
    Flickr8kPaths(tiny_flickr8k).image("train_2.jpg").write_bytes(b"")
    with pytest.raises(DatasetError, match=r"'train_2\.jpg': file is empty"):
        load_flickr8k(tiny_flickr8k)


def test_image_without_five_captions_is_reported(tiny_flickr8k: Path) -> None:
    captions = Flickr8kPaths(tiny_flickr8k).captions
    lines = captions.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith("val_0.jpg#4")]
    captions.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"4 caption\(s\) instead of 5"):
        load_flickr8k(tiny_flickr8k)


def test_captioned_images_outside_splits_are_tolerated(tiny_flickr8k: Path) -> None:
    # The real token file captions one image that is absent from the archive.
    with Flickr8kPaths(tiny_flickr8k).captions.open("a", encoding="utf-8") as stream:
        stream.write("ghost.jpg#0\tAn image that does not exist .\n")
    assert "ghost.jpg" in load_flickr8k(tiny_flickr8k).captions


def test_download_file_copies_and_skips_existing(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "nested" / "copy.bin"
    assert download_file(source.as_uri(), destination) == destination
    assert destination.read_bytes() == b"payload"

    source.write_bytes(b"changed")
    download_file(source.as_uri(), destination)
    assert destination.read_bytes() == b"payload"


def test_failed_download_leaves_no_file(tmp_path: Path) -> None:
    destination = tmp_path / "copy.bin"
    with pytest.raises(OSError, match="absent"):
        download_file((tmp_path / "absent.bin").as_uri(), destination)
    assert list(tmp_path.iterdir()) == []


def test_extract_archive_skips_macos_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("kept.txt", "content")
        bundle.writestr("__MACOSX/._kept.txt", "metadata")
    extract_archive(archive, tmp_path / "out")
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["kept.txt"]


def test_extract_archive_refuses_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "content")
    with pytest.raises(DatasetError, match="unsafe path"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_archive_rejects_corrupted_archive(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(DatasetError, match="not a valid zip"):
        extract_archive(archive, tmp_path / "out")


@pytest.mark.parametrize("keep_archives", [False, True])
def test_download_flickr8k_end_to_end(tmp_path: Path, keep_archives: bool) -> None:
    images_url, text_url = _archives(tmp_path)
    root = tmp_path / "dataset"
    dataset = download_flickr8k(
        root, images_url=images_url, text_url=text_url, keep_archives=keep_archives
    )
    assert dataset.splits.sizes() == {"train": 4, "val": 2, "test": 2}
    assert (root / "images.zip").exists() is keep_archives
    assert (root / "text.zip").exists() is keep_archives


def test_download_flickr8k_does_nothing_when_dataset_is_present(tiny_flickr8k: Path) -> None:
    dataset = download_flickr8k(
        tiny_flickr8k, images_url="file:///nonexistent.zip", text_url="file:///nonexistent.zip"
    )
    assert len(dataset.captions) == 8
