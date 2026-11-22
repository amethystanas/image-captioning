"""Extraction and on-disk caching of CLIP image embeddings.

CLIP stays frozen, so each image is encoded exactly once and training reads
the cached embeddings. Extraction is resumable: progress is checkpointed to a
``.partial`` file that an interrupted run picks up again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from clipcap.utils import get_logger

logger = get_logger(__name__)

_FORMAT_VERSION = 1


class FeatureCacheError(RuntimeError):
    """Raised when a feature cache is unreadable or does not match the request."""


class ImageEncoder(Protocol):
    """Anything that turns a batch of images into fixed-size embeddings."""

    @property
    def name(self) -> str:
        """Identifier of the encoder, stored alongside the features."""
        ...

    def encode(self, images: Sequence[Image.Image]) -> Tensor:
        """Return a ``(len(images), embedding_dim)`` float tensor."""
        ...


class ClipImageEncoder:
    """Frozen CLIP vision tower producing projected image embeddings."""

    def __init__(self, model_name: str, device: torch.device | str = "cpu") -> None:
        """Load the CLIP checkpoint ``model_name`` on ``device`` and freeze it."""
        from transformers import CLIPModel, CLIPProcessor

        self._name = model_name
        self._device = torch.device(device)
        self._processor: Any = CLIPProcessor.from_pretrained(model_name)
        model: Any = CLIPModel.from_pretrained(model_name)
        self._model: Any = model.to(self._device).eval()
        self._model.requires_grad_(False)

    @property
    def name(self) -> str:
        """Hugging Face identifier of the CLIP checkpoint."""
        return self._name

    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image]) -> Tensor:
        """Return the projected CLIP embeddings of ``images`` on CPU."""
        inputs = self._processor(images=list(images), return_tensors="pt")
        output: Any = self._model.get_image_features(
            pixel_values=inputs["pixel_values"].to(self._device)
        )
        # transformers < 5 returns the tensor, >= 5 wraps it in a model output.
        features: Tensor = output if isinstance(output, Tensor) else output.pooler_output
        return features.float().cpu()


@dataclass(frozen=True)
class FeatureSet:
    """Embeddings of a list of images, row ``i`` belonging to ``names[i]``."""

    names: tuple[str, ...]
    embeddings: Tensor
    encoder_name: str
    normalized: bool

    def __post_init__(self) -> None:
        """Check that names and embeddings are consistent."""
        if self.embeddings.ndim != 2:
            raise FeatureCacheError(
                f"embeddings must be 2-dimensional, got shape {tuple(self.embeddings.shape)}"
            )
        if len(self.names) != self.embeddings.shape[0]:
            raise FeatureCacheError(
                f"{len(self.names)} name(s) for {self.embeddings.shape[0]} embedding(s)"
            )
        if len(set(self.names)) != len(self.names):
            raise FeatureCacheError("image names must be unique")

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.names)

    @property
    def dim(self) -> int:
        """Dimension of one embedding."""
        return int(self.embeddings.shape[1])

    def index(self) -> dict[str, int]:
        """Map each image name to its row."""
        return {name: row for row, name in enumerate(self.names)}

    def save(self, path: Path) -> None:
        """Write the feature set atomically to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        payload = {
            "version": _FORMAT_VERSION,
            "names": list(self.names),
            "embeddings": self.embeddings,
            "encoder_name": self.encoder_name,
            "normalized": self.normalized,
        }
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> FeatureSet:
        """Read a feature set written by :meth:`save`."""
        if not path.is_file():
            raise FileNotFoundError(f"feature file not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["version"] != _FORMAT_VERSION:
                raise FeatureCacheError(f"{path}: unsupported format version {payload['version']}")
            return cls(
                names=tuple(payload["names"]),
                embeddings=payload["embeddings"],
                encoder_name=payload["encoder_name"],
                normalized=payload["normalized"],
            )
        except FeatureCacheError:
            raise
        except Exception as exc:
            raise FeatureCacheError(f"{path} is not a valid feature file: {exc!r}") from exc


def extract_features(
    encoder: ImageEncoder,
    images_dir: Path,
    names: Sequence[str],
    cache_path: Path,
    *,
    batch_size: int = 64,
    normalize: bool = True,
    save_every: int = 20,
    progress: bool = False,
) -> FeatureSet:
    """Return the embeddings of ``names``, computing only what is not cached.

    A complete cache at ``cache_path`` is returned as is. Otherwise images are
    encoded by batches, progress being saved every ``save_every`` batches so
    that an interrupted run resumes where it stopped. A cache built with a
    different encoder, normalisation or image list raises
    :class:`FeatureCacheError` rather than being silently reused or overwritten.
    """
    if batch_size <= 0 or save_every <= 0:
        raise ValueError("batch_size and save_every must be positive")
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("image names must be unique")
    if not names:
        raise ValueError("no image to encode")

    if cache_path.is_file():
        cached = FeatureSet.load(cache_path)
        _check_compatible(cached, encoder, normalize, cache_path)
        if cached.names != names:
            raise FeatureCacheError(
                f"{cache_path} was built for a different image list; "
                "delete it or use another cache path"
            )
        logger.info("loaded %d cached embeddings from %s", len(cached), cache_path)
        return cached

    partial_path = cache_path.with_name(cache_path.name + ".partial")
    done_names: list[str] = []
    chunks: list[Tensor] = []
    if partial_path.is_file():
        partial = FeatureSet.load(partial_path)
        _check_compatible(partial, encoder, normalize, partial_path)
        if partial.names != names[: len(partial)]:
            raise FeatureCacheError(
                f"{partial_path} does not match the requested image list; delete it to restart"
            )
        done_names, chunks = list(partial.names), [partial.embeddings]
        logger.info("resuming extraction after %d image(s)", len(done_names))

    def snapshot() -> FeatureSet:
        return FeatureSet(tuple(done_names), torch.cat(chunks), encoder.name, normalize)

    starts = range(len(done_names), len(names), batch_size)
    for batch_number, start in enumerate(
        tqdm(starts, desc="CLIP features", disable=not progress), start=1
    ):
        batch_names = names[start : start + batch_size]
        embeddings = encoder.encode([_open_image(images_dir / name) for name in batch_names])
        if embeddings.shape[0] != len(batch_names):
            raise FeatureCacheError(
                f"encoder returned {embeddings.shape[0]} embedding(s) for "
                f"{len(batch_names)} image(s)"
            )
        embeddings = embeddings.detach().float().cpu()
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        chunks.append(embeddings)
        done_names.extend(batch_names)
        if batch_number % save_every == 0 and len(done_names) < len(names):
            snapshot().save(partial_path)

    features = snapshot()
    features.save(cache_path)
    partial_path.unlink(missing_ok=True)
    logger.info("saved %d embeddings to %s", len(features), cache_path)
    return features


def _check_compatible(
    cached: FeatureSet, encoder: ImageEncoder, normalize: bool, path: Path
) -> None:
    if cached.encoder_name != encoder.name:
        raise FeatureCacheError(
            f"{path} was built with encoder '{cached.encoder_name}', not '{encoder.name}'"
        )
    if cached.normalized != normalize:
        raise FeatureCacheError(
            f"{path} was built with normalize={cached.normalized}, not normalize={normalize}"
        )


def _open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            converted: Image.Image = image.convert("RGB")
            return converted
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise FeatureCacheError(f"cannot read image {path}: {exc!r}") from exc
