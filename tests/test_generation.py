import pytest
import torch
from torch import Tensor

from clipcap.generation import greedy_decode
from clipcap.models.clipcap import ClipCaptionModel
from tests.helpers import make_tiny_model

CLIP_DIM = 8
EOS = 0


@pytest.fixture(scope="module")
def trained() -> tuple[ClipCaptionModel, Tensor, list[list[int]]]:
    """A fine-tuned tiny model that has memorised one caption per image."""
    torch.manual_seed(0)
    model = make_tiny_model(freeze_gpt2=False).train()
    generator = torch.Generator().manual_seed(1)
    embeddings = torch.nn.functional.normalize(
        torch.randn(3, CLIP_DIM, generator=generator), dim=-1
    )
    captions = [[5, 6, 7, 8], [9, 10], [11, 12, 13]]
    tokens = torch.zeros(3, 5, dtype=torch.long)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    for row, caption in enumerate(captions):
        tokens[row, : len(caption)] = torch.tensor(caption)
        mask[row, : len(caption) + 1] = True  # caption followed by end-of-sequence
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    for _ in range(120):
        optimizer.zero_grad()
        model(embeddings, tokens, mask).loss.backward()
        optimizer.step()
    return model.eval(), embeddings, captions


def _untrained(embeddings_seed: int = 0) -> tuple[ClipCaptionModel, Tensor]:
    generator = torch.Generator().manual_seed(embeddings_seed)
    return make_tiny_model(), torch.randn(4, CLIP_DIM, generator=generator)


def _manual_greedy(model: ClipCaptionModel, embedding: Tensor, steps: int) -> list[int]:
    """Reference implementation: recompute the full sequence at every step, no cache."""
    inputs = model.prefix(embedding.unsqueeze(0))
    tokens: list[int] = []
    for _ in range(steps):
        token = int(model.gpt2(inputs_embeds=inputs).logits[0, -1].argmax())
        if token == EOS:
            break
        tokens.append(token)
        inputs = torch.cat([inputs, model.embed_tokens(torch.tensor([[token]]))], dim=1)
    return tokens


def test_greedy_reproduces_memorised_captions(
    trained: tuple[ClipCaptionModel, Tensor, list[list[int]]],
) -> None:
    model, embeddings, captions = trained
    assert greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=10) == captions


def test_greedy_matches_uncached_reference() -> None:
    model, embeddings = _untrained()
    decoded = greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=12)
    with torch.no_grad():
        expected = [_manual_greedy(model, embedding, 12) for embedding in embeddings]
    assert decoded == expected


def test_greedy_is_deterministic_and_independent_of_batching() -> None:
    model, embeddings = _untrained()
    together = greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=8)
    assert greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=8) == together
    alone = [
        greedy_decode(model, embeddings[row : row + 1], eos_token_id=EOS, max_new_tokens=8)[0]
        for row in range(len(embeddings))
    ]
    assert alone == together


@pytest.mark.parametrize("max_new_tokens", [1, 4, 9])
def test_greedy_respects_the_length_limit_and_never_emits_eos(max_new_tokens: int) -> None:
    model, embeddings = _untrained()
    decoded = greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=max_new_tokens)
    assert len(decoded) == len(embeddings)
    for tokens in decoded:
        assert len(tokens) <= max_new_tokens
        assert EOS not in tokens


def test_greedy_stops_each_caption_at_its_own_eos(
    trained: tuple[ClipCaptionModel, Tensor, list[list[int]]],
) -> None:
    model, embeddings, captions = trained
    decoded = greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=20)
    assert [len(tokens) for tokens in decoded] == [len(caption) for caption in captions]


def test_generation_restores_training_mode_and_builds_no_graph() -> None:
    model, embeddings = _untrained()
    model.train()
    greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=3)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=3)
    assert not model.training


@pytest.mark.parametrize(
    ("max_new_tokens", "message"),
    [(0, "must be positive"), (30, "exceeds the 29 positions left")],
)
def test_invalid_length_budget_is_rejected(max_new_tokens: int, message: str) -> None:
    model, embeddings = _untrained()
    with pytest.raises(ValueError, match=message):
        greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=max_new_tokens)
