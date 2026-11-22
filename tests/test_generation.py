import itertools
import math

import pytest
import torch
from torch import Tensor

from clipcap.config import GenerationConfig
from clipcap.generation import (
    beam_search,
    beam_search_hypotheses,
    generate,
    greedy_decode,
    nucleus_probabilities,
    nucleus_sample,
)
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


def _sequence_log_prob(
    model: ClipCaptionModel, embedding: Tensor, tokens: tuple[int, ...], terminated: bool
) -> float:
    """Log-probability of ``tokens`` (plus end-of-sequence if ``terminated``)."""
    targets = [*tokens, EOS] if terminated else list(tokens)
    inputs = model.prefix(embedding.unsqueeze(0))
    if len(targets) > 1:
        history = torch.tensor([targets[:-1]])
        inputs = torch.cat([inputs, model.embed_tokens(history)], dim=1)
    logits = model.gpt2(inputs_embeds=inputs).logits[0, model.prefix_length - 1 :]
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(sum(log_probs[step, token] for step, token in enumerate(targets)))


def test_beam_size_one_is_greedy() -> None:
    model, embeddings = _untrained()
    greedy = greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=10)
    assert (
        beam_search(model, embeddings, eos_token_id=EOS, max_new_tokens=10, beam_size=1) == greedy
    )


def test_beam_search_reproduces_memorised_captions(
    trained: tuple[ClipCaptionModel, Tensor, list[list[int]]],
) -> None:
    model, embeddings, captions = trained
    assert (
        beam_search(model, embeddings, eos_token_id=EOS, max_new_tokens=10, beam_size=4) == captions
    )


@pytest.mark.parametrize("length_penalty", [0.0, 1.0])
def test_reported_log_prob_and_score_are_those_of_the_returned_tokens(
    length_penalty: float,
) -> None:
    model, embeddings = _untrained()
    hypotheses = beam_search_hypotheses(
        model,
        embeddings,
        eos_token_id=EOS,
        max_new_tokens=6,
        beam_size=3,
        length_penalty=length_penalty,
    )
    assert len(hypotheses) == len(embeddings)
    with torch.no_grad():
        for embedding, found in zip(embeddings, hypotheses, strict=True):
            expected = _sequence_log_prob(model, embedding, found.tokens, found.terminated)
            assert found.log_prob == pytest.approx(expected, abs=1e-4)
            length = len(found.tokens) + int(found.terminated)
            assert found.score == pytest.approx(expected / length**length_penalty, abs=1e-4)
            assert EOS not in found.tokens
            assert len(found.tokens) <= 6


def test_exhaustive_beam_finds_the_most_probable_caption() -> None:
    """With a beam as wide as the search space, beam search is exact."""
    vocabulary, steps = 5, 3
    from clipcap.models.mapper import MLPMapper
    from tests.helpers import TINY_GPT_DIM, make_tiny_gpt2

    torch.manual_seed(3)
    gpt2 = make_tiny_gpt2(seed=3, vocab_size=vocabulary)
    for parameter in gpt2.parameters():  # sharpen the near-uniform random distribution
        parameter.data.mul_(4.0)
    model = ClipCaptionModel(gpt2, MLPMapper(CLIP_DIM, TINY_GPT_DIM, 3)).eval()
    embedding = torch.randn(CLIP_DIM)

    with torch.no_grad():
        candidates: dict[tuple[tuple[int, ...], bool], float] = {}
        non_eos = range(1, vocabulary)
        for length in range(steps + 1):
            for tokens in itertools.product(non_eos, repeat=length):
                if length < steps:
                    candidates[(tokens, True)] = _sequence_log_prob(model, embedding, tokens, True)
                else:
                    candidates[(tokens, False)] = _sequence_log_prob(
                        model, embedding, tokens, False
                    )
    (best_tokens, best_terminated), best_log_prob = max(candidates.items(), key=lambda kv: kv[1])
    assert math.isclose(sum(math.exp(value) for value in candidates.values()), 1.0, abs_tol=1e-4)

    found = beam_search_hypotheses(
        model,
        embedding.unsqueeze(0),
        eos_token_id=EOS,
        max_new_tokens=steps,
        beam_size=vocabulary**steps,
        length_penalty=0.0,
    )[0]
    assert found.tokens == best_tokens
    assert found.terminated is best_terminated
    assert found.log_prob == pytest.approx(best_log_prob, abs=1e-4)


def test_beam_search_is_independent_of_batching() -> None:
    model, embeddings = _untrained()
    together = beam_search(model, embeddings, eos_token_id=EOS, max_new_tokens=6, beam_size=3)
    alone = [
        beam_search(
            model, embeddings[row : row + 1], eos_token_id=EOS, max_new_tokens=6, beam_size=3
        )[0]
        for row in range(len(embeddings))
    ]
    assert alone == together


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beam_size": 0}, "beam_size must be positive"),
        ({"length_penalty": -0.5}, "length_penalty must be non-negative"),
        ({"max_new_tokens": 0}, "max_new_tokens must be positive"),
    ],
)
def test_beam_search_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    model, embeddings = _untrained()
    arguments = {"max_new_tokens": 5, **kwargs}
    with pytest.raises(ValueError, match=message):
        beam_search(model, embeddings, eos_token_id=EOS, **arguments)  # type: ignore[arg-type]


HAND_LOGITS = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))


@pytest.mark.parametrize(
    ("top_p", "expected"),
    [
        (1.0, [0.15, 0.5, 0.05, 0.3]),
        (0.9, [0.15 / 0.95, 0.5 / 0.95, 0.0, 0.3 / 0.95]),
        (0.7, [0.0, 0.625, 0.0, 0.375]),
        (0.49, [0.0, 1.0, 0.0, 0.0]),
        (0.51, [0.0, 0.625, 0.0, 0.375]),
        (1e-6, [0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_nucleus_keeps_the_smallest_set_reaching_top_p(top_p: float, expected: list[float]) -> None:
    probabilities = nucleus_probabilities(HAND_LOGITS, top_p=top_p)
    assert torch.allclose(probabilities, torch.tensor([expected]), atol=1e-5)


def test_temperature_sharpens_or_flattens_the_distribution() -> None:
    cold = nucleus_probabilities(HAND_LOGITS, top_p=1.0, temperature=0.05)
    hot = nucleus_probabilities(HAND_LOGITS, top_p=1.0, temperature=100.0)
    assert cold[0, 1] > 0.999
    assert torch.allclose(hot, torch.full((1, 4), 0.25), atol=0.02)


def test_nucleus_handles_batches_row_by_row() -> None:
    logits = torch.cat([HAND_LOGITS, HAND_LOGITS.flip(-1)])
    probabilities = nucleus_probabilities(logits, top_p=0.7)
    assert torch.allclose(probabilities[1], probabilities[0].flip(-1), atol=1e-6)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))


def test_sampling_is_reproducible_with_a_seeded_generator() -> None:
    model, embeddings = _untrained()

    def sample(seed: int) -> list[list[int]]:
        return nucleus_sample(
            model,
            embeddings,
            eos_token_id=EOS,
            max_new_tokens=10,
            generator=torch.Generator().manual_seed(seed),
        )

    assert sample(0) == sample(0)
    assert sample(0) != sample(1)


@pytest.mark.parametrize("kwargs", [{"top_p": 1e-6}, {"top_p": 1.0, "temperature": 1e-4}])
def test_degenerate_sampling_is_greedy(kwargs: dict[str, float]) -> None:
    model, embeddings = _untrained()
    sampled = nucleus_sample(
        model,
        embeddings,
        eos_token_id=EOS,
        max_new_tokens=10,
        generator=torch.Generator().manual_seed(0),
        **kwargs,
    )
    assert sampled == greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=10)


def test_sampled_captions_respect_length_limit_and_never_contain_eos() -> None:
    model, embeddings = _untrained()
    sampled = nucleus_sample(
        model,
        embeddings,
        eos_token_id=EOS,
        max_new_tokens=7,
        top_p=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert all(len(tokens) <= 7 and EOS not in tokens for tokens in sampled)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_p": 0.0}, "top_p must be in"),
        ({"top_p": 1.5}, "top_p must be in"),
        ({"temperature": 0.0}, "temperature must be positive"),
    ],
)
def test_nucleus_rejects_invalid_arguments(kwargs: dict[str, float], message: str) -> None:
    model, embeddings = _untrained()
    with pytest.raises(ValueError, match=message):
        nucleus_sample(model, embeddings, eos_token_id=EOS, max_new_tokens=5, **kwargs)


def test_generate_dispatches_on_the_configured_strategy() -> None:
    model, embeddings = _untrained()
    greedy = generate(
        model, embeddings, GenerationConfig(strategy="greedy", max_new_tokens=6), eos_token_id=EOS
    )
    assert greedy == greedy_decode(model, embeddings, eos_token_id=EOS, max_new_tokens=6)

    beam_config = GenerationConfig(
        strategy="beam", max_new_tokens=6, beam_size=3, length_penalty=0.5
    )
    assert generate(model, embeddings, beam_config, eos_token_id=EOS) == beam_search(
        model, embeddings, eos_token_id=EOS, max_new_tokens=6, beam_size=3, length_penalty=0.5
    )

    nucleus_config = GenerationConfig(strategy="nucleus", max_new_tokens=6, top_p=0.8)
    sampled = generate(
        model,
        embeddings,
        nucleus_config,
        eos_token_id=EOS,
        generator=torch.Generator().manual_seed(4),
    )
    assert sampled == nucleus_sample(
        model,
        embeddings,
        eos_token_id=EOS,
        max_new_tokens=6,
        top_p=0.8,
        generator=torch.Generator().manual_seed(4),
    )
