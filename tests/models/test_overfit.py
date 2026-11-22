"""Sanity check: the model must be able to memorise a handful of examples."""

import pytest
import torch
from torch import Tensor

from clipcap.config import ModelConfig
from clipcap.models.clipcap import ClipCaptionModel
from clipcap.models.mapper import build_mapper
from tests.helpers import TINY_GPT_DIM, TINY_VOCAB, make_tiny_model

CLIP_DIM = 8


def _memorisation_batch() -> tuple[Tensor, Tensor, Tensor]:
    """Four distinct images, each with its own caption of a different length."""
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.nn.functional.normalize(
        torch.randn(4, CLIP_DIM, generator=generator), dim=-1
    )
    lengths = [6, 5, 4, 3]
    tokens = torch.zeros(4, 6, dtype=torch.long)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    for row, length in enumerate(lengths):
        tokens[row, : length - 1] = torch.randint(1, TINY_VOCAB, (length - 1,), generator=generator)
        mask[row, :length] = True  # the last real token is end-of-sequence (id 0)
    return embeddings, tokens, mask


def _fit(model: ClipCaptionModel, batch: tuple[Tensor, Tensor, Tensor], steps: int) -> float:
    """Train on ``batch`` for ``steps`` updates and return the evaluation loss."""
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        model(*batch).loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return float(model(*batch).loss.item())


def test_fine_tuned_model_memorises_four_examples() -> None:
    torch.manual_seed(0)
    model = make_tiny_model(freeze_gpt2=False)
    batch = _memorisation_batch()
    initial = model(*batch).loss.item()

    assert _fit(model, batch, steps=100) < 0.05 < initial

    _, tokens, mask = batch
    with torch.no_grad():
        predictions = model(*batch).logits.argmax(dim=-1)
    assert torch.equal(predictions[mask], tokens[mask])


@pytest.mark.parametrize("mapper_type", ["mlp", "transformer"])
def test_mapper_alone_memorises_four_examples_with_a_frozen_language_model(
    mapper_type: str,
) -> None:
    """A frozen GPT-2 that already models the captions is steered by the prefix only.

    A randomly initialised GPT-2 outputs near-uniform logits whatever its
    input, so it is first trained as part of another model, which stands in
    for pretraining, and only then frozen and given a fresh mapper.
    """
    torch.manual_seed(0)
    batch = _memorisation_batch()
    pretrained = make_tiny_model(freeze_gpt2=False)
    _fit(pretrained, batch, steps=100)

    mapper = build_mapper(
        ModelConfig(
            mapper_type=mapper_type, prefix_length=3, mapper_num_layers=1, mapper_num_heads=2
        ),
        CLIP_DIM,
        TINY_GPT_DIM,
    )
    model = ClipCaptionModel(pretrained.gpt2, mapper, freeze_gpt2=True)
    frozen_weights = {name: value.clone() for name, value in model.gpt2.state_dict().items()}
    initial = model.eval()(*batch).loss.item()

    assert _fit(model, batch, steps=150) < 0.1 < initial
    for name, value in model.gpt2.state_dict().items():
        assert torch.equal(value, frozen_weights[name]), name
