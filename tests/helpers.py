"""Network-free stand-ins for the pretrained components."""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from clipcap.config import ModelConfig
from clipcap.data.features import FeatureSet
from clipcap.models.clipcap import ClipCaptionModel
from clipcap.models.mapper import build_mapper


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


TINY_VOCAB = 40
TINY_GPT_DIM = 16


def make_tiny_gpt2(seed: int = 0, vocab_size: int = TINY_VOCAB) -> GPT2LMHeadModel:
    """Return a randomly initialised two-layer GPT-2, built without network access."""
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_embd=TINY_GPT_DIM,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(config)


def make_tiny_model(
    *,
    clip_dim: int = 8,
    prefix_length: int = 3,
    freeze_gpt2: bool = True,
    mapper_type: str = "mlp",
    seed: int = 0,
) -> ClipCaptionModel:
    """Return a tiny captioning model in evaluation mode."""
    gpt2 = make_tiny_gpt2(seed)
    config = ModelConfig(
        mapper_type=mapper_type,
        prefix_length=prefix_length,
        freeze_gpt2=freeze_gpt2,
        mapper_num_layers=1,
        mapper_num_heads=2,
    )
    mapper = build_mapper(config, clip_dim, TINY_GPT_DIM)
    model = ClipCaptionModel(gpt2, mapper, freeze_gpt2=freeze_gpt2)
    return model.eval()
