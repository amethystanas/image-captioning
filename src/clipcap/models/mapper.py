"""Mapping networks: from one CLIP embedding to a prefix of GPT-2 embeddings.

A mapper turns a ``(batch, clip_dim)`` image embedding into a
``(batch, prefix_length, gpt_dim)`` sequence that GPT-2 reads as if it were
the embeddings of ``prefix_length`` tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from clipcap.config import ModelConfig


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


class TransformerMapper(nn.Module):
    """Transformer mapper of ClipCap, designed for a frozen GPT-2.

    The CLIP embedding is projected to ``prefix_length`` tokens, concatenated
    with ``prefix_length`` learned constant tokens and passed through a
    Transformer encoder. The outputs at the constant positions form the
    prefix: the constants act as queries that attend to the image tokens.
    """

    def __init__(
        self,
        clip_dim: int,
        gpt_dim: int,
        prefix_length: int,
        *,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Build the mapper; ``gpt_dim`` must be divisible by ``num_heads``."""
        super().__init__()
        _check_positive(
            clip_dim=clip_dim,
            gpt_dim=gpt_dim,
            prefix_length=prefix_length,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        if gpt_dim % num_heads != 0:
            raise ValueError(f"gpt_dim ({gpt_dim}) must be divisible by num_heads ({num_heads})")
        self.clip_dim = clip_dim
        self.gpt_dim = gpt_dim
        self.prefix_length = prefix_length
        self.project = nn.Linear(clip_dim, prefix_length * gpt_dim)
        self.queries = nn.Parameter(torch.randn(prefix_length, gpt_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=gpt_dim,
            nhead=num_heads,
            dim_feedforward=2 * gpt_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Map ``(batch, clip_dim)`` embeddings to ``(batch, prefix_length, gpt_dim)``."""
        _check_input(embeddings, self.clip_dim)
        batch_size = embeddings.shape[0]
        image_tokens = self.project(embeddings).view(batch_size, self.prefix_length, self.gpt_dim)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        encoded: Tensor = self.encoder(torch.cat([image_tokens, queries], dim=1))
        return encoded[:, self.prefix_length :]


def build_mapper(config: ModelConfig, clip_dim: int, gpt_dim: int) -> MLPMapper | TransformerMapper:
    """Instantiate the mapper described by ``config``."""
    if config.mapper_type == "mlp":
        return MLPMapper(
            clip_dim,
            gpt_dim,
            config.prefix_length,
            hidden_ratio=config.mapper_hidden_ratio,
            dropout=config.mapper_dropout,
        )
    if config.mapper_type == "transformer":
        return TransformerMapper(
            clip_dim,
            gpt_dim,
            config.prefix_length,
            num_layers=config.mapper_num_layers,
            num_heads=config.mapper_num_heads,
            dropout=config.mapper_dropout,
        )
    raise ValueError(f"unknown mapper type '{config.mapper_type}'")  # pragma: no cover


def _check_positive(**values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _check_input(embeddings: Tensor, clip_dim: int) -> None:
    if embeddings.ndim != 2 or embeddings.shape[1] != clip_dim:
        raise ValueError(
            f"expected embeddings of shape (batch, {clip_dim}), got {tuple(embeddings.shape)}"
        )
