"""Mapping networks: from one CLIP embedding to a prefix of GPT-2 embeddings.

A mapper turns a ``(batch, clip_dim)`` image embedding into a
``(batch, prefix_length, gpt_dim)`` sequence that GPT-2 reads as if it were
the embeddings of ``prefix_length`` tokens.
"""

from __future__ import annotations

from torch import Tensor, nn


class MLPMapper(nn.Module):
    """Two-layer perceptron producing the whole prefix at once.

    This is the light mapper of ClipCap; it is meant to be trained together
    with a fine-tuned GPT-2, but also works with a frozen one.
    """

    def __init__(
        self,
        clip_dim: int,
        gpt_dim: int,
        prefix_length: int,
        *,
        hidden_ratio: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        """Build the mapper.

        The hidden layer has ``hidden_ratio * prefix_length * gpt_dim`` units.
        """
        super().__init__()
        _check_positive(clip_dim=clip_dim, gpt_dim=gpt_dim, prefix_length=prefix_length)
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive")
        self.clip_dim = clip_dim
        self.gpt_dim = gpt_dim
        self.prefix_length = prefix_length
        output_dim = prefix_length * gpt_dim
        hidden_dim = max(1, int(hidden_ratio * output_dim))
        self.network = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """Map ``(batch, clip_dim)`` embeddings to ``(batch, prefix_length, gpt_dim)``."""
        _check_input(embeddings, self.clip_dim)
        prefix: Tensor = self.network(embeddings)
        return prefix.view(-1, self.prefix_length, self.gpt_dim)


def _check_positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _check_input(embeddings: Tensor, clip_dim: int) -> None:
    if embeddings.ndim != 2 or embeddings.shape[1] != clip_dim:
        raise ValueError(
            f"expected embeddings of shape (batch, {clip_dim}), got {tuple(embeddings.shape)}"
        )
