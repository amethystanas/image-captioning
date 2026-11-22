"""Generate captions for a set of images and score them."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from tqdm import tqdm

from clipcap.config import GenerationConfig
from clipcap.data.features import FeatureSet
from clipcap.evaluation.metrics import CaptionScores, compute_metrics
from clipcap.generation import generate
from clipcap.models.clipcap import ClipCaptionModel


class TextDecoder(Protocol):
    """The part of a Hugging Face tokenizer needed to turn ids back into text."""

    def decode(self, token_ids: list[int]) -> str:
        """Return the text of ``token_ids``."""
        ...


@dataclass(frozen=True)
class EvaluationResult:
    """Scores of a model on a split, with every generated caption."""

    split: str
    scores: CaptionScores
    generation: GenerationConfig
    captions: dict[str, str]
    references: dict[str, list[str]]

    @property
    def num_images(self) -> int:
        """Number of images that were captioned."""
        return len(self.captions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the evaluation."""
        return {
            "split": self.split,
            "num_images": self.num_images,
            "scores": self.scores.to_dict(),
            "generation": dataclasses.asdict(self.generation),
            "captions": [
                {"image": image, "caption": caption, "references": self.references[image]}
                for image, caption in self.captions.items()
            ],
        }

    def save(self, path: str | Path) -> None:
        """Write the evaluation as JSON, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8")


def generate_captions(
    model: ClipCaptionModel,
    features: FeatureSet,
    names: Sequence[str],
    tokenizer: TextDecoder,
    generation: GenerationConfig,
    *,
    eos_token_id: int,
    batch_size: int = 32,
    seed: int = 0,
    progress: bool = False,
) -> dict[str, str]:
    """Return one generated caption per image of ``names``, in order.

    The model is used on whatever device it currently lives on. ``seed`` only
    matters for sampling strategies and makes them reproducible.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not names:
        raise ValueError("no image to caption")
    rows = features.index()
    missing = [name for name in names if name not in rows]
    if missing:
        raise KeyError(f"no cached embedding for {len(missing)} image(s), e.g. '{missing[0]}'")

    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    captions: dict[str, str] = {}
    starts = range(0, len(names), batch_size)
    for start in tqdm(starts, desc="captioning", disable=not progress):
        batch_names = names[start : start + batch_size]
        embeddings = features.embeddings[[rows[name] for name in batch_names]].to(device)
        sequences = generate(
            model, embeddings, generation, eos_token_id=eos_token_id, generator=generator
        )
        for name, token_ids in zip(batch_names, sequences, strict=True):
            captions[name] = tokenizer.decode(token_ids).strip()
    return captions


def evaluate_model(
    model: ClipCaptionModel,
    features: FeatureSet,
    references: Mapping[str, Sequence[str]],
    names: Sequence[str],
    tokenizer: TextDecoder,
    generation: GenerationConfig,
    *,
    eos_token_id: int,
    split: str = "test",
    batch_size: int = 32,
    seed: int = 0,
    meteor: bool = False,
    progress: bool = False,
) -> EvaluationResult:
    """Caption the images of ``names`` and score them against ``references``."""
    without_reference = [name for name in names if not references.get(name)]
    if without_reference:
        raise KeyError(
            f"no reference caption for {len(without_reference)} image(s), "
            f"e.g. '{without_reference[0]}'"
        )
    captions = generate_captions(
        model,
        features,
        names,
        tokenizer,
        generation,
        eos_token_id=eos_token_id,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
    )
    kept_references = {name: list(references[name]) for name in captions}
    scores = compute_metrics(kept_references, captions, meteor=meteor)
    return EvaluationResult(
        split=split,
        scores=scores,
        generation=generation,
        captions=captions,
        references=kept_references,
    )
