"""Prefix captioning model: a mapper feeding a GPT-2 language model.

The image prefix produced by the mapper is concatenated in front of the
caption token embeddings; GPT-2 is trained (or kept frozen) to continue the
prefix with the caption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from clipcap.config import ModelConfig
from clipcap.models.mapper import build_mapper

_IGNORE_INDEX = -100


@dataclass(frozen=True)
class CaptionOutput:
    """Result of a training forward pass.

    ``logits`` is ``(batch, length, vocabulary)``: ``logits[:, t]`` scores
    caption token ``t`` given the image prefix and tokens ``0 .. t-1``.
    """

    loss: Tensor
    logits: Tensor


class ClipCaptionModel(nn.Module):
    """Mapper + GPT-2, trained with a next-token loss on the caption only."""

    def __init__(self, gpt2: nn.Module, mapper: nn.Module, *, freeze_gpt2: bool = True) -> None:
        """Assemble the model from a GPT-2 language model and a mapper.

        ``gpt2`` must be a ``GPT2LMHeadModel`` (or expose the same interface)
        and ``mapper`` must output prefixes of GPT-2's embedding size.
        """
        super().__init__()
        language_model: Any = gpt2
        prefix_mapper: Any = mapper
        gpt_dim = int(language_model.config.n_embd)
        if int(prefix_mapper.gpt_dim) != gpt_dim:
            raise ValueError(
                f"mapper outputs embeddings of size {prefix_mapper.gpt_dim}, "
                f"GPT-2 expects {gpt_dim}"
            )
        self.gpt2: Any = language_model
        self.mapper = mapper
        self.freeze_gpt2 = freeze_gpt2
        self.prefix_length = int(prefix_mapper.prefix_length)
        if freeze_gpt2:
            self.gpt2.requires_grad_(False)
            self.gpt2.eval()

    @classmethod
    def from_config(cls, config: ModelConfig, clip_dim: int) -> ClipCaptionModel:
        """Load the pretrained GPT-2 named in ``config`` and build its mapper."""
        from transformers import GPT2LMHeadModel

        gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_name)
        mapper = build_mapper(config, clip_dim, int(gpt2.config.n_embd))
        return cls(gpt2, mapper, freeze_gpt2=config.freeze_gpt2)

    @property
    def max_positions(self) -> int:
        """Longest sequence (prefix included) GPT-2 can read."""
        return int(self.gpt2.config.n_positions)

    def train(self, mode: bool = True) -> ClipCaptionModel:
        """Switch mode, keeping a frozen GPT-2 in evaluation mode (no dropout)."""
        super().train(mode)
        if self.freeze_gpt2:
            self.gpt2.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the parameters an optimiser should update."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """Return the GPT-2 input embeddings of ``tokens``."""
        embedded: Tensor = self.gpt2.get_input_embeddings()(tokens)
        return embedded

    def prefix(self, embeddings: Tensor) -> Tensor:
        """Map image embeddings to ``(batch, prefix_length, gpt_dim)`` prefixes."""
        prefix: Tensor = self.mapper(embeddings)
        return prefix

    def forward(self, embeddings: Tensor, tokens: Tensor, mask: Tensor) -> CaptionOutput:
        """Compute the caption loss.

        ``tokens`` and ``mask`` are ``(batch, length)``; positions where
        ``mask`` is ``False`` are padding and contribute neither to the loss
        nor, through attention, to the other positions.
        """
        if tokens.shape != mask.shape or tokens.ndim != 2:
            raise ValueError(
                f"tokens {tuple(tokens.shape)} and mask {tuple(mask.shape)} "
                "must share a (batch, length) shape"
            )
        if embeddings.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"{embeddings.shape[0]} image embedding(s) for {tokens.shape[0]} caption(s)"
            )
        length = self.prefix_length + tokens.shape[1]
        if length > self.max_positions:
            raise ValueError(
                f"prefix + caption length {length} exceeds GPT-2's limit of {self.max_positions}"
            )

        prefix = self.prefix(embeddings)
        inputs = torch.cat([prefix, self.embed_tokens(tokens)], dim=1)
        prefix_mask = torch.ones(
            tokens.shape[0], self.prefix_length, dtype=torch.bool, device=mask.device
        )
        attention_mask = torch.cat([prefix_mask, mask.bool()], dim=1)
        all_logits: Tensor = self.gpt2(
            inputs_embeds=inputs, attention_mask=attention_mask.long()
        ).logits

        # The output at position p predicts the input at p + 1: the last prefix
        # position predicts the first caption token, and the output at the last
        # caption token has no target.
        logits = all_logits[:, self.prefix_length - 1 : -1]
        labels = tokens.masked_fill(~mask.bool(), _IGNORE_INDEX)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=_IGNORE_INDEX
        )
        return CaptionOutput(loss=loss, logits=logits)
