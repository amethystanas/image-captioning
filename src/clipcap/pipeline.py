"""End-to-end stages: feature extraction, training, evaluation and captioning.

Each stage takes an :class:`ExperimentConfig` and a :class:`Components`
factory. The default factory loads the pretrained CLIP and GPT-2; tests
substitute tiny stand-ins so that every stage runs without network access.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from clipcap.config import ExperimentConfig, ModelConfig
from clipcap.data.dataset import CaptionDataset
from clipcap.data.download import Flickr8k, load_flickr8k
from clipcap.data.features import (
    ClipImageEncoder,
    FeatureSet,
    ImageEncoder,
    extract_features,
    open_image,
)
from clipcap.data.splits import SPLIT_NAMES
from clipcap.evaluation.evaluate import EvaluationResult, evaluate_model
from clipcap.generation import generate
from clipcap.models.clipcap import ClipCaptionModel
from clipcap.training.tracking import RunTracker, prepare_run_directory
from clipcap.training.trainer import BEST_CHECKPOINT, Trainer, TrainingState, load_model_weights
from clipcap.utils import get_logger, resolve_device, seed_everything

logger = get_logger(__name__)


class PipelineError(RuntimeError):
    """Raised when a stage is run before the stage it depends on."""


def _load_tokenizer(config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.gpt2_name)


@dataclass(frozen=True)
class Components:
    """Factories of the pretrained building blocks.

    ``tokenizer`` must return an object with ``encode``, ``decode`` and
    ``eos_token_id``, like a Hugging Face GPT-2 tokenizer.
    """

    encoder: Callable[[ModelConfig, torch.device], ImageEncoder] = lambda config, device: (
        ClipImageEncoder(config.clip_name, device)
    )
    tokenizer: Callable[[ModelConfig], Any] = _load_tokenizer
    model: Callable[[ModelConfig, int], ClipCaptionModel] = ClipCaptionModel.from_config


def feature_path(config: ExperimentConfig, split: str) -> Path:
    """Path of the cached embeddings of ``split``."""
    return Path(config.data.features_dir) / f"{split}.pt"


def _load_features(config: ExperimentConfig, split: str) -> FeatureSet:
    path = feature_path(config, split)
    if not path.is_file():
        raise PipelineError(
            f"no cached features for split '{split}' at {path}: run 'extract' first"
        )
    return FeatureSet.load(path)


def run_extract(
    config: ExperimentConfig,
    components: Components | None = None,
    *,
    batch_size: int = 64,
    progress: bool = False,
) -> dict[str, FeatureSet]:
    """Encode the images of every split with CLIP and cache the embeddings."""
    components = components or Components()
    dataset = load_flickr8k(config.data.root)
    encoder = components.encoder(config.model, resolve_device(config.training.device))
    features = {}
    for split in SPLIT_NAMES:
        features[split] = extract_features(
            encoder,
            dataset.paths.images_dir,
            dataset.splits[split],
            feature_path(config, split),
            batch_size=batch_size,
            normalize=True,
            progress=progress,
        )
        logger.info("split '%s': %d embeddings", split, len(features[split]))
    return features


def _loader(
    config: ExperimentConfig, dataset: Flickr8k, split: str, tokenizer: Any, *, shuffle: bool
) -> tuple[DataLoader[Any], int]:
    features = _load_features(config, split)
    examples = CaptionDataset(
        features,
        dataset.captions,
        dataset.splits[split],
        tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        max_tokens=config.data.max_caption_tokens,
    )
    loader: DataLoader[Any] = DataLoader(
        examples,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        collate_fn=examples.collate,
    )
    return loader, examples.embedding_dim


def run_train(
    config: ExperimentConfig, components: Components | None = None, *, resume: bool = False
) -> TrainingState:
    """Train the model described by ``config`` and return the final state.

    With ``resume``, training continues from the last checkpoint of the run
    directory when there is one.
    """
    components = components or Components()
    seed_everything(config.training.seed)
    dataset = load_flickr8k(config.data.root)
    tokenizer = components.tokenizer(config.model)
    train_loader, clip_dim = _loader(config, dataset, "train", tokenizer, shuffle=True)
    val_loader, _ = _loader(config, dataset, "val", tokenizer, shuffle=False)
    model = components.model(config.model, clip_dim)

    directory = prepare_run_directory(config)
    with RunTracker(directory) as tracker:
        trainer = Trainer(model, train_loader, val_loader, config.training, tracker=tracker)
        if trainer.last_checkpoint.is_file() and not resume:
            raise PipelineError(
                f"{directory} already contains a checkpoint: pass --resume to continue it "
                "or choose another training.output_dir"
            )
        if resume and trainer.resume():
            tracker.restore_history(trainer.state.history)
        return trainer.fit()


def _trained_model(
    config: ExperimentConfig, components: Components, clip_dim: int, checkpoint: Path | None
) -> ClipCaptionModel:
    path = checkpoint or Path(config.training.output_dir) / BEST_CHECKPOINT
    model = components.model(config.model, clip_dim)
    state = load_model_weights(path, model)
    logger.info("loaded %s (epoch %d, val loss %.4f)", path, state.epoch, state.best_val_loss)
    return model.to(resolve_device(config.training.device)).eval()


def run_evaluate(
    config: ExperimentConfig,
    components: Components | None = None,
    *,
    split: str = "test",
    checkpoint: Path | None = None,
    output: Path | None = None,
    meteor: bool = False,
    progress: bool = False,
) -> EvaluationResult:
    """Score a trained model on ``split`` and save the result as JSON.

    The checkpoint defaults to the best one of the run directory and the
    report to ``eval_<split>_<strategy>.json`` in the same directory.
    """
    if split not in SPLIT_NAMES:
        raise PipelineError(f"unknown split '{split}', expected one of {SPLIT_NAMES}")
    components = components or Components()
    dataset = load_flickr8k(config.data.root)
    features = _load_features(config, split)
    tokenizer = components.tokenizer(config.model)
    model = _trained_model(config, components, features.dim, checkpoint)
    result = evaluate_model(
        model,
        features,
        dataset.captions,
        dataset.splits[split],
        tokenizer,
        config.generation,
        eos_token_id=tokenizer.eos_token_id,
        split=split,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
        meteor=meteor,
        progress=progress,
    )
    if output is None:
        name = f"eval_{split}_{config.generation.strategy}.json"
        output = Path(config.training.output_dir) / name
    result.save(output)
    logger.info("saved evaluation to %s", output)
    return result


def run_caption(
    config: ExperimentConfig,
    images: Sequence[Path],
    components: Components | None = None,
    *,
    checkpoint: Path | None = None,
) -> dict[str, str]:
    """Caption arbitrary image files with a trained model."""
    if not images:
        raise PipelineError("no image to caption")
    components = components or Components()
    device = resolve_device(config.training.device)
    encoder = components.encoder(config.model, device)
    embeddings = encoder.encode([open_image(path) for path in images]).float()
    # Cached training features are L2-normalised: new images must be too.
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    tokenizer = components.tokenizer(config.model)
    model = _trained_model(config, components, int(embeddings.shape[1]), checkpoint)
    sequences = generate(
        model,
        embeddings.to(device),
        config.generation,
        eos_token_id=tokenizer.eos_token_id,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    return {
        str(path): tokenizer.decode(tokens).strip()
        for path, tokens in zip(images, sequences, strict=True)
    }
