import json
from pathlib import Path

import pytest

from clipcap.data.splits import (
    SplitError,
    Splits,
    load_karpathy_splits,
    load_split_files,
    make_random_splits,
)


def _karpathy(tmp_path: Path, entries: object) -> Path:
    path = tmp_path / "dataset_flickr8k.json"
    path.write_text(json.dumps({"images": entries}), encoding="utf-8")
    return path


def _entries(counts: dict[str, int]) -> list[dict[str, str]]:
    return [
        {"filename": f"{split}_{index}.jpg", "split": split}
        for split, count in counts.items()
        for index in range(count)
    ]


def test_splits_accessors() -> None:
    splits = Splits(train=("a", "b"), val=("c",), test=("d",))
    assert splits["train"] == ("a", "b")
    assert splits.sizes() == {"train": 2, "val": 1, "test": 1}
    assert [name for name, _ in splits.items()] == ["train", "val", "test"]
    with pytest.raises(KeyError, match="unknown split"):
        splits["dev"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"train": ("a",), "val": (), "test": ("b",)}, "'val' is empty"),
        ({"train": ("a", "a"), "val": ("b",), "test": ("c",)}, "duplicate"),
        ({"train": ("a",), "val": ("b",), "test": ("a",)}, "both 'train' and 'test'"),
    ],
)
def test_inconsistent_splits_are_rejected(kwargs: dict[str, tuple[str, ...]], message: str) -> None:
    with pytest.raises(SplitError, match=message):
        Splits(**kwargs)


def test_check_against_reports_unknown_images() -> None:
    splits = Splits(train=("a",), val=("b",), test=("c",))
    splits.check_against(["a", "b", "c", "extra"])
    with pytest.raises(SplitError, match="'test' references 1 unknown"):
        splits.check_against(["a", "b"])


def test_load_karpathy_splits(tmp_path: Path) -> None:
    splits = load_karpathy_splits(_karpathy(tmp_path, _entries({"train": 6, "val": 2, "test": 3})))
    assert splits.sizes() == {"train": 6, "val": 2, "test": 3}
    assert splits.test == ("test_0.jpg", "test_1.jpg", "test_2.jpg")


def test_karpathy_restval_is_folded_into_train(tmp_path: Path) -> None:
    entries = _entries({"train": 2, "restval": 3, "val": 1, "test": 1})
    assert load_karpathy_splits(_karpathy(tmp_path, entries)).sizes()["train"] == 5


def test_karpathy_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Karpathy split file"):
        load_karpathy_splits(tmp_path / "absent.json")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "not a Karpathy split file"),
        ('{"pictures": []}', "not a Karpathy split file"),
        ("[1, 2]", "not a Karpathy split file"),
        ('{"images": [{"filename": "a.jpg"}]}', "entry 0 lacks"),
        ('{"images": ["a.jpg"]}', "entry 0 lacks"),
        ('{"images": [{"filename": "a.jpg", "split": "dev"}]}', "unknown split 'dev'"),
    ],
)
def test_malformed_karpathy_files_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SplitError, match=message):
        load_karpathy_splits(path)


def test_load_split_files(tmp_path: Path) -> None:
    for name, content in {
        "train": "a.jpg\nb.jpg\n",
        "val": "c.jpg\n\n",
        "test": " d.jpg \n",
    }.items():
        (tmp_path / f"{name}.txt").write_text(content, encoding="utf-8")
    splits = load_split_files(tmp_path / "train.txt", tmp_path / "val.txt", tmp_path / "test.txt")
    assert splits == Splits(train=("a.jpg", "b.jpg"), val=("c.jpg",), test=("d.jpg",))


def test_load_split_files_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="split file not found"):
        load_split_files(tmp_path / "a", tmp_path / "b", tmp_path / "c")


def test_random_splits_have_requested_sizes_and_cover_everything() -> None:
    images = [f"{index}.jpg" for index in range(100)]
    splits = make_random_splits(images, val_size=10, test_size=20)
    assert splits.sizes() == {"train": 70, "val": 10, "test": 20}
    assert sorted(splits.train + splits.val + splits.test) == sorted(images)


def test_random_splits_are_deterministic_and_order_independent() -> None:
    images = [f"{index}.jpg" for index in range(50)]
    first = make_random_splits(images, val_size=5, test_size=5, seed=7)
    assert make_random_splits(reversed(images), val_size=5, test_size=5, seed=7) == first
    assert make_random_splits(images, val_size=5, test_size=5, seed=8) != first


@pytest.mark.parametrize(
    ("val_size", "test_size", "message"),
    [(0, 1, "must be positive"), (2, 2, "cannot provide")],
)
def test_random_splits_reject_impossible_sizes(val_size: int, test_size: int, message: str) -> None:
    with pytest.raises(SplitError, match=message):
        make_random_splits(["a", "b", "c", "d"], val_size=val_size, test_size=test_size)
