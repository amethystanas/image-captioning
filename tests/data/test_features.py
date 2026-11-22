from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor

from clipcap.data.features import (
    ClipImageEncoder,
    FeatureCacheError,
    FeatureSet,
    extract_features,
)
from tests.conftest import TINY_SPLITS

NAMES = TINY_SPLITS["train"] + TINY_SPLITS["val"] + TINY_SPLITS["test"]
DIM = 6


class FakeEncoder:
    """Deterministic encoder built on the mean colour of each image."""

    def __init__(self, name: str = "fake-clip", fail_after: int | None = None) -> None:
        self._name = name
        self.encoded = 0
        self.calls = 0
        self._fail_after = fail_after

    @property
    def name(self) -> str:
        return self._name

    def encode(self, images: Sequence[Image.Image]) -> Tensor:
        if self._fail_after is not None and self.calls >= self._fail_after:
            raise KeyboardInterrupt
        self.calls += 1
        self.encoded += len(images)
        rows = []
        for image in images:
            red, green, blue = image.resize((1, 1)).getpixel((0, 0))  # type: ignore[misc]
            rows.append([red, green, blue, red + green, green + blue, 1.0])
        return torch.tensor(rows, dtype=torch.float64)


@pytest.fixture
def images_dir(tiny_flickr8k: Path) -> Path:
    return tiny_flickr8k / "Flicker8k_Dataset"


def test_extraction_shapes_order_and_normalisation(images_dir: Path, tmp_path: Path) -> None:
    features = extract_features(FakeEncoder(), images_dir, NAMES, tmp_path / "f.pt", batch_size=3)
    assert features.names == NAMES
    assert features.embeddings.shape == (len(NAMES), DIM)
    assert features.embeddings.dtype == torch.float32
    assert features.dim == DIM
    assert torch.allclose(features.embeddings.norm(dim=-1), torch.ones(len(NAMES)), atol=1e-5)
    assert features.index()["val_0.jpg"] == 4


def test_embeddings_do_not_depend_on_batch_size(images_dir: Path, tmp_path: Path) -> None:
    one = extract_features(FakeEncoder(), images_dir, NAMES, tmp_path / "a.pt", batch_size=1)
    many = extract_features(FakeEncoder(), images_dir, NAMES, tmp_path / "b.pt", batch_size=64)
    assert torch.equal(one.embeddings, many.embeddings)


def test_normalisation_can_be_disabled(images_dir: Path, tmp_path: Path) -> None:
    features = extract_features(
        FakeEncoder(), images_dir, NAMES, tmp_path / "f.pt", normalize=False
    )
    assert features.normalized is False
    assert features.embeddings.norm(dim=-1).min() > 1.5


def test_complete_cache_is_reused_without_encoding(images_dir: Path, tmp_path: Path) -> None:
    cache = tmp_path / "nested" / "f.pt"
    first = extract_features(FakeEncoder(), images_dir, NAMES, cache)
    encoder = FakeEncoder()
    second = extract_features(encoder, images_dir, NAMES, cache)
    assert encoder.encoded == 0
    assert second.names == first.names
    assert torch.equal(second.embeddings, first.embeddings)
    assert [path.name for path in cache.parent.iterdir()] == ["f.pt"]


def test_interrupted_extraction_resumes_from_partial(images_dir: Path, tmp_path: Path) -> None:
    cache = tmp_path / "f.pt"
    with pytest.raises(KeyboardInterrupt):
        extract_features(
            FakeEncoder(fail_after=2), images_dir, NAMES, cache, batch_size=2, save_every=1
        )
    assert not cache.exists()
    assert len(FeatureSet.load(tmp_path / "f.pt.partial")) == 4

    encoder = FakeEncoder()
    resumed = extract_features(encoder, images_dir, NAMES, cache, batch_size=2, save_every=1)
    assert encoder.encoded == len(NAMES) - 4
    assert not (tmp_path / "f.pt.partial").exists()

    reference = extract_features(FakeEncoder(), images_dir, NAMES, tmp_path / "ref.pt")
    assert resumed.names == reference.names
    assert torch.equal(resumed.embeddings, reference.embeddings)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"encoder": FakeEncoder("other-clip")}, "built with encoder 'fake-clip'"),
        ({"normalize": False}, "normalize=True, not normalize=False"),
        ({"names": NAMES[:3]}, "different image list"),
    ],
)
def test_incompatible_cache_is_refused(
    images_dir: Path, tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    cache = tmp_path / "f.pt"
    extract_features(FakeEncoder(), images_dir, NAMES, cache)
    arguments: dict[str, object] = {"encoder": FakeEncoder(), "names": NAMES, **kwargs}
    with pytest.raises(FeatureCacheError, match=message):
        extract_features(images_dir=images_dir, cache_path=cache, **arguments)  # type: ignore[arg-type]


def test_incompatible_partial_is_refused(images_dir: Path, tmp_path: Path) -> None:
    cache = tmp_path / "f.pt"
    with pytest.raises(KeyboardInterrupt):
        extract_features(
            FakeEncoder(fail_after=1), images_dir, NAMES, cache, batch_size=2, save_every=1
        )
    with pytest.raises(FeatureCacheError, match="does not match the requested image list"):
        extract_features(FakeEncoder(), images_dir, tuple(reversed(NAMES)), cache)


def test_corrupted_cache_is_reported(images_dir: Path, tmp_path: Path) -> None:
    cache = tmp_path / "f.pt"
    cache.write_bytes(b"garbage")
    with pytest.raises(FeatureCacheError, match="not a valid feature file"):
        extract_features(FakeEncoder(), images_dir, NAMES, cache)


def test_unsupported_cache_version_is_reported(tmp_path: Path) -> None:
    cache = tmp_path / "f.pt"
    torch.save({"version": 99}, cache)
    with pytest.raises(FeatureCacheError, match="unsupported format version 99"):
        FeatureSet.load(cache)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="feature file not found"):
        FeatureSet.load(tmp_path / "absent.pt")


def test_missing_image_is_reported(images_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_features(FakeEncoder(), images_dir, ("absent.jpg",), tmp_path / "f.pt")


def test_unreadable_image_is_reported(images_dir: Path, tmp_path: Path) -> None:
    (images_dir / "broken.jpg").write_bytes(b"not an image")
    with pytest.raises(FeatureCacheError, match="cannot read image"):
        extract_features(FakeEncoder(), images_dir, ("broken.jpg",), tmp_path / "f.pt")


def test_encoder_returning_wrong_batch_size_is_reported(images_dir: Path, tmp_path: Path) -> None:
    class Broken(FakeEncoder):
        def encode(self, images: Sequence[Image.Image]) -> Tensor:
            return super().encode(images)[:1]

    with pytest.raises(FeatureCacheError, match="returned 1 embedding"):
        extract_features(Broken(), images_dir, NAMES, tmp_path / "f.pt", batch_size=4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"names": ()}, "no image to encode"),
        ({"names": ("a.jpg", "a.jpg")}, "must be unique"),
        ({"names": NAMES, "batch_size": 0}, "must be positive"),
        ({"names": NAMES, "save_every": 0}, "must be positive"),
    ],
)
def test_invalid_arguments(
    images_dir: Path, tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_features(FakeEncoder(), images_dir, cache_path=tmp_path / "f.pt", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("names", "embeddings", "message"),
    [
        (("a",), torch.zeros(3), "2-dimensional"),
        (("a",), torch.zeros(2, 3), r"1 name\(s\) for 2 embedding"),
        (("a", "a"), torch.zeros(2, 3), "must be unique"),
    ],
)
def test_inconsistent_feature_set_is_rejected(
    names: tuple[str, ...], embeddings: Tensor, message: str
) -> None:
    with pytest.raises(FeatureCacheError, match=message):
        FeatureSet(names, embeddings, "fake-clip", True)


@pytest.mark.slow
def test_real_clip_encoder(images_dir: Path, tmp_path: Path) -> None:
    encoder = ClipImageEncoder("openai/clip-vit-base-patch32")
    features = extract_features(encoder, images_dir, NAMES[:3], tmp_path / "f.pt")
    assert features.embeddings.shape == (3, 512)
    assert torch.isfinite(features.embeddings).all()
    assert torch.allclose(features.embeddings.norm(dim=-1), torch.ones(3), atol=1e-4)
