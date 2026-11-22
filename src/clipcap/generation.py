"""Caption decoding strategies.

Every strategy takes image embeddings and returns, for each image, the
generated token ids without the end-of-sequence token.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

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
