"""Caption decoding strategies.

Every strategy takes image embeddings and returns, for each image, the
generated token ids without the end-of-sequence token.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from clipcap.models.clipcap import ClipCaptionModel


@contextmanager
def _evaluation_mode(model: ClipCaptionModel) -> Iterator[None]:
    """Run a block in evaluation mode and restore the previous mode afterwards."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _check_budget(model: ClipCaptionModel, max_new_tokens: int) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    available = model.max_positions - model.prefix_length
    if max_new_tokens > available:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} exceeds the {available} positions left "
            "after the prefix"
        )


def _next_token_logits(model: ClipCaptionModel, inputs: Tensor) -> Tensor:
    """Return the ``(batch, vocabulary)`` logits of the token following ``inputs``."""
    logits: Tensor = model.gpt2(inputs_embeds=inputs).logits
    return logits[:, -1]


def _decode_stepwise(
    model: ClipCaptionModel,
    embeddings: Tensor,
    select: Callable[[Tensor], Tensor],
    *,
    eos_token_id: int,
    max_new_tokens: int,
) -> list[list[int]]:
    """Generate one token per step for the whole batch.

    ``select`` maps ``(batch, vocabulary)`` logits to ``(batch,)`` token ids.
    The whole sequence is re-read at every step rather than relying on GPT-2's
    key/value cache, whose interface differs between transformers versions;
    captions are short enough for the extra cost to be negligible.
    """
    _check_budget(model, max_new_tokens)
    batch_size = embeddings.shape[0]
    sequences: list[list[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    with torch.no_grad(), _evaluation_mode(model):
        inputs = model.prefix(embeddings)
        for _ in range(max_new_tokens):
            next_tokens = select(_next_token_logits(model, inputs))
            for row, token in enumerate(next_tokens.tolist()):
                if finished[row]:
                    continue
                if token == eos_token_id:
                    finished[row] = True
                else:
                    sequences[row].append(token)
            if all(finished):
                break
            inputs = torch.cat([inputs, model.embed_tokens(next_tokens).unsqueeze(1)], dim=1)
    return sequences


def greedy_decode(
    model: ClipCaptionModel, embeddings: Tensor, *, eos_token_id: int, max_new_tokens: int = 30
) -> list[list[int]]:
    """Pick the most likely token at every step.

    Generation of a caption stops at the end-of-sequence token or after
    ``max_new_tokens`` tokens.
    """
    return _decode_stepwise(
        model,
        embeddings,
        lambda logits: logits.argmax(dim=-1),
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )


@dataclass(frozen=True)
class BeamHypothesis:
    """A caption found by beam search.

    ``log_prob`` is the log-probability of the tokens, end-of-sequence
    included when the caption was terminated; ``score`` is ``log_prob``
    divided by ``length ** length_penalty`` and is what hypotheses are ranked by.
    """

    tokens: tuple[int, ...]
    log_prob: float
    score: float
    terminated: bool


def beam_search_hypotheses(
    model: ClipCaptionModel,
    embeddings: Tensor,
    *,
    eos_token_id: int,
    max_new_tokens: int = 30,
    beam_size: int = 5,
    length_penalty: float = 1.0,
) -> list[BeamHypothesis]:
    """Return the best-scoring hypothesis of each image.

    At every step the ``beam_size`` most likely continuations are kept; a
    hypothesis that emits end-of-sequence is set aside and frees no slot, so
    the search narrows as captions terminate. ``length_penalty`` of 0 ranks by
    raw log-probability (favouring short captions), 1 by mean token
    log-probability.
    """
    _check_budget(model, max_new_tokens)
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    if length_penalty < 0:
        raise ValueError("length_penalty must be non-negative")
    with torch.no_grad(), _evaluation_mode(model):
        prefixes = model.prefix(embeddings)
        return [
            _beam_search_one(model, prefix, eos_token_id, max_new_tokens, beam_size, length_penalty)
            for prefix in prefixes
        ]


def beam_search(
    model: ClipCaptionModel,
    embeddings: Tensor,
    *,
    eos_token_id: int,
    max_new_tokens: int = 30,
    beam_size: int = 5,
    length_penalty: float = 1.0,
) -> list[list[int]]:
    """Return the tokens of the best beam-search hypothesis of each image."""
    hypotheses = beam_search_hypotheses(
        model,
        embeddings,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )
    return [list(hypothesis.tokens) for hypothesis in hypotheses]


def _beam_search_one(
    model: ClipCaptionModel,
    prefix: Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> BeamHypothesis:
    """Beam search for one image, ``prefix`` being ``(prefix_length, gpt_dim)``."""

    def hypothesis(tokens: tuple[int, ...], log_prob: float, terminated: bool) -> BeamHypothesis:
        length = len(tokens) + int(terminated)
        return BeamHypothesis(
            tokens, log_prob, log_prob / max(length, 1) ** length_penalty, terminated
        )

    live: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    done: list[BeamHypothesis] = []
    width = beam_size
    for _ in range(max_new_tokens):
        inputs = prefix.unsqueeze(0).expand(len(live), -1, -1)
        if live[0][0]:
            tokens = torch.tensor([beam for beam, _ in live], device=prefix.device)
            inputs = torch.cat([inputs, model.embed_tokens(tokens)], dim=1)
        log_probs = torch.log_softmax(_next_token_logits(model, inputs), dim=-1)
        totals = torch.tensor(
            [total for _, total in live], device=prefix.device, dtype=log_probs.dtype
        )
        candidates = (totals.unsqueeze(1) + log_probs).flatten()
        best = torch.topk(candidates, min(width, candidates.numel()))
        vocabulary = log_probs.shape[1]

        next_live: list[tuple[tuple[int, ...], float]] = []
        for total, flat_index in zip(best.values.tolist(), best.indices.tolist(), strict=True):
            beam, token = live[flat_index // vocabulary][0], flat_index % vocabulary
            if token == eos_token_id:
                done.append(hypothesis(beam, total, True))
                width -= 1
            else:
                next_live.append(((*beam, token), total))
        live = next_live
        if not live:
            break

    done.extend(hypothesis(beam, total, False) for beam, total in live)
    return max(done, key=lambda item: item.score)
