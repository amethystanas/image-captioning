"""Network-free stand-ins for the pretrained components."""

from __future__ import annotations

import torch

from clipcap.data.features import FeatureSet


class WordTokenizer:
    """Whitespace tokenizer with a growing vocabulary; id 0 is end-of-sequence."""

    eos_token_id = 0

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = ["<eos>"]

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._words)
                self._words.append(word)
            ids.append(self._ids[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._words[index] for index in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._words)


def make_features(names: tuple[str, ...], dim: int = 8, seed: int = 0) -> FeatureSet:
    """Return random unit-norm embeddings for ``names``."""
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.nn.functional.normalize(
        torch.randn(len(names), dim, generator=generator), dim=-1
    )
    return FeatureSet(names, embeddings, "fake-clip", True)
