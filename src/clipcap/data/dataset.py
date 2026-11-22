"""Training examples: one cached image embedding paired with one caption."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from clipcap.data.features import FeatureSet


class Tokenizer(Protocol):
    """The part of a Hugging Face tokenizer the dataset relies on."""

    def encode(self, text: str) -> list[int]:
        """Return the token ids of ``text``."""
        ...


@dataclass(frozen=True)
class CaptionBatch:
    """A padded batch of examples.

    ``tokens`` is ``(batch, length)`` and ends each caption with the
    end-of-sequence token; ``mask`` is ``True`` on real tokens and ``False``
    on padding.
    """

    embeddings: Tensor
    tokens: Tensor
    mask: Tensor

    def to(self, device: torch.device | str) -> CaptionBatch:
        """Return a copy of the batch on ``device``."""
        return CaptionBatch(
            embeddings=self.embeddings.to(device),
            tokens=self.tokens.to(device),
            mask=self.mask.to(device),
        )

    def __len__(self) -> int:
        """Return the number of examples."""
        return int(self.tokens.shape[0])


class CaptionDataset(Dataset[tuple[Tensor, Tensor]]):
    """All ``(image embedding, caption tokens)`` pairs of a set of images.

    Captions are tokenised once at construction. Each one is truncated to
    ``max_tokens`` tokens, end-of-sequence included, so that every caption
    ends with the end-of-sequence token even when truncated.
    """

    def __init__(
        self,
        features: FeatureSet,
        captions: Mapping[str, Sequence[str]],
        names: Sequence[str],
        tokenizer: Tokenizer,
        *,
        eos_token_id: int,
        max_tokens: int = 40,
    ) -> None:
        """Build the examples of the images called ``names``."""
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2 (one token plus end-of-sequence)")
        rows = features.index()
        self._embeddings = features.embeddings
        self._rows: list[int] = []
        sequences: list[list[int]] = []
        self._image_names: list[str] = []
        for name in names:
            if name not in rows:
                raise KeyError(f"no cached embedding for image '{name}'")
            if not captions.get(name):
                raise KeyError(f"no caption for image '{name}'")
            for caption in captions[name]:
                ids = tokenizer.encode(caption)[: max_tokens - 1]
                if not ids:
                    raise ValueError(f"caption of '{name}' has no token: {caption!r}")
                self._rows.append(rows[name])
                sequences.append([*ids, eos_token_id])
                self._image_names.append(name)
        if not self._rows:
            raise ValueError("the dataset is empty")
        # One padded tensor rather than one tensor per caption: worker processes
        # started with "spawn" or "forkserver" receive each tensor through its
        # own file descriptor, and tens of thousands of them exceed the limit.
        self._lengths = [len(sequence) for sequence in sequences]
        self._tokens = torch.full(
            (len(sequences), max(self._lengths)), eos_token_id, dtype=torch.long
        )
        for row, sequence in enumerate(sequences):
            self._tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        """Return the number of (image, caption) pairs."""
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return the embedding and token ids of example ``index``."""
        return self._embeddings[self._rows[index]], self._tokens[index, : self._lengths[index]]

    def image_name(self, index: int) -> str:
        """Return the name of the image behind example ``index``."""
        return self._image_names[index]

    @property
    def embedding_dim(self) -> int:
        """Dimension of the image embeddings."""
        return int(self._embeddings.shape[1])

    def collate(self, examples: Sequence[tuple[Tensor, Tensor]]) -> CaptionBatch:
        """Collate examples, padding with this dataset's end-of-sequence token."""
        return collate_captions(examples, pad_token_id=self.eos_token_id)


def collate_captions(
    examples: Sequence[tuple[Tensor, Tensor]], *, pad_token_id: int
) -> CaptionBatch:
    """Stack embeddings and right-pad token sequences to the longest one.

    GPT-2 has no padding token, so ``pad_token_id`` is usually its
    end-of-sequence id; padded positions are flagged in the mask and must be
    excluded from the loss.
    """
    if not examples:
        raise ValueError("cannot collate an empty batch")
    embeddings = torch.stack([embedding for embedding, _ in examples])
    lengths = [int(tokens.shape[0]) for _, tokens in examples]
    tokens = torch.full((len(examples), max(lengths)), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(examples), max(lengths)), dtype=torch.bool)
    for row, (_, sequence) in enumerate(examples):
        tokens[row, : lengths[row]] = sequence
        mask[row, : lengths[row]] = True
    return CaptionBatch(embeddings=embeddings, tokens=tokens, mask=mask)


CollateFn = Callable[[Sequence[tuple[Tensor, Tensor]]], CaptionBatch]
