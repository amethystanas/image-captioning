"""Command-line interface: ``clipcap <command> --config <file.yaml>``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from clipcap import __version__
from clipcap.config import DECODING_STRATEGIES, ConfigError, ExperimentConfig
from clipcap.data.download import DatasetError, download_flickr8k
from clipcap.data.features import FeatureCacheError
from clipcap.data.flickr8k import CaptionFormatError
from clipcap.data.splits import SPLIT_NAMES, SplitError
from clipcap.evaluation.report import ReportError, results_table
from clipcap.pipeline import (
    Components,
    PipelineError,
    run_caption,
    run_evaluate,
    run_extract,
    run_train,
)
from clipcap.training.tracking import RunDirectoryError
from clipcap.training.trainer import CheckpointError
from clipcap.utils import setup_logging

# Problems a user can fix (bad path, missing stage, wrong file): reported in
# one line instead of a traceback.
_USER_ERRORS = (
    ConfigError,
    DatasetError,
    CaptionFormatError,
    SplitError,
    FeatureCacheError,
    CheckpointError,
    RunDirectoryError,
    PipelineError,
    ReportError,
    FileNotFoundError,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``clipcap`` command."""
    parser = argparse.ArgumentParser(
        prog="clipcap", description="CLIP + GPT-2 prefix captioning on Flickr8k."
    )
    parser.add_argument("--version", action="version", version=f"clipcap {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log debug messages")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--config", type=Path, required=True, help="experiment YAML file")
        return sub

    command("download", "Download Flickr8k into data.root and verify it.")

    extract = command("extract", "Encode every image with CLIP and cache the embeddings.")
    extract.add_argument("--batch-size", type=int, default=64, help="images per CLIP batch")

    train = command("train", "Train the captioning model.")
    train.add_argument("--resume", action="store_true", help="continue from the last checkpoint")

    evaluate = command("evaluate", "Score a trained model on a split.")
    evaluate.add_argument("--split", choices=SPLIT_NAMES, default="test")
    evaluate.add_argument("--checkpoint", type=Path, help="defaults to the run's best checkpoint")
    evaluate.add_argument("--output", type=Path, help="where to write the JSON report")
    evaluate.add_argument("--meteor", action="store_true", help="also compute METEOR (needs Java)")
    evaluate.add_argument(
        "--strategy",
        choices=DECODING_STRATEGIES,
        help="decoding strategy, overriding generation.strategy of the configuration",
    )

    caption = command("caption", "Caption image files with a trained model.")
    caption.add_argument("images", type=Path, nargs="+", help="image files")
    caption.add_argument("--checkpoint", type=Path, help="defaults to the run's best checkpoint")

    report = commands.add_parser(
        "report",
        help="Print a Markdown table comparing evaluation reports.",
        description="Print a Markdown table comparing evaluation reports.",
    )
    report.add_argument("reports", type=Path, nargs="+", help="eval_*.json files")
    return parser


def main(argv: Sequence[str] | None = None, components: Components | None = None) -> int:
    """Run the command line and return the process exit code.

    ``components`` replaces the pretrained models, which tests use to run
    every command offline.
    """
    arguments = build_parser().parse_args(argv)
    setup_logging("DEBUG" if arguments.verbose else "INFO")
    try:
        if arguments.command == "report":
            print(results_table(arguments.reports), end="")
        else:
            config = ExperimentConfig.from_yaml(arguments.config)
            _dispatch(arguments, config, components)
    except _USER_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _dispatch(
    arguments: argparse.Namespace, config: ExperimentConfig, components: Components | None
) -> None:
    if arguments.command == "download":
        dataset = download_flickr8k(config.data.root)
        print(f"Flickr8k ready under {config.data.root}: {dataset.splits.sizes()}")
    elif arguments.command == "extract":
        features = run_extract(
            config, components, batch_size=arguments.batch_size, progress=sys.stderr.isatty()
        )
        for split, feature_set in features.items():
            print(f"{split}: {len(feature_set)} embeddings of dimension {feature_set.dim}")
    elif arguments.command == "train":
        state = run_train(config, components, resume=arguments.resume)
        print(
            f"trained {state.epoch} epoch(s); best validation loss "
            f"{state.best_val_loss:.4f} at epoch {state.best_epoch}"
        )
    elif arguments.command == "evaluate":
        if arguments.strategy is not None:
            generation = dataclasses.replace(config.generation, strategy=arguments.strategy)
            config = dataclasses.replace(config, generation=generation)
        result = run_evaluate(
            config,
            components,
            split=arguments.split,
            checkpoint=arguments.checkpoint,
            output=arguments.output,
            meteor=arguments.meteor,
            progress=sys.stderr.isatty(),
        )
        print(f"{result.split} split, {result.num_images} images")
        for name, value in result.scores.to_dict().items():
            print(f"  {name:<8} {100 * value:6.2f}")
    else:
        captions = run_caption(
            config, arguments.images, components, checkpoint=arguments.checkpoint
        )
        for path, caption in captions.items():
            print(f"{path}\t{caption}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
