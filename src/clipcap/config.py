"""Typed experiment configuration loaded from YAML files.

Every section is a frozen dataclass that validates itself on construction,
so an invalid experiment fails at load time rather than hours into a run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

MAPPER_TYPES = ("mlp", "transformer")
DECODING_STRATEGIES = ("greedy", "beam", "nucleus")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or holds invalid values."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class DataConfig:
    """Location of the dataset and of the cached CLIP features."""

    root: Path = Path("data/flickr8k")
    features_dir: Path = Path("data/features")
    max_caption_tokens: int = 40
    num_workers: int = 2

    def __post_init__(self) -> None:
        """Validate field values."""
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "features_dir", Path(self.features_dir))
        _require(self.max_caption_tokens > 0, "data.max_caption_tokens must be positive")
        _require(self.num_workers >= 0, "data.num_workers must be non-negative")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the prefix captioning model."""

    clip_name: str = "openai/clip-vit-base-patch32"
    gpt2_name: str = "gpt2"
    mapper_type: str = "mlp"
    prefix_length: int = 10
    freeze_gpt2: bool = True
    mapper_hidden_ratio: float = 0.5
    mapper_num_layers: int = 8
    mapper_num_heads: int = 8
    mapper_dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate field values."""
        _require(
            self.mapper_type in MAPPER_TYPES,
            f"model.mapper_type must be one of {MAPPER_TYPES}, got '{self.mapper_type}'",
        )
        _require(self.prefix_length > 0, "model.prefix_length must be positive")
        _require(self.mapper_hidden_ratio > 0, "model.mapper_hidden_ratio must be positive")
        _require(self.mapper_num_layers > 0, "model.mapper_num_layers must be positive")
        _require(self.mapper_num_heads > 0, "model.mapper_num_heads must be positive")
        _require(0.0 <= self.mapper_dropout < 1.0, "model.mapper_dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation hyper-parameters and run bookkeeping."""

    output_dir: Path = Path("outputs/default")
    epochs: int = 10
    batch_size: int = 40
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    patience: int = 3
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        """Validate field values."""
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        _require(self.epochs > 0, "training.epochs must be positive")
        _require(self.batch_size > 0, "training.batch_size must be positive")
        _require(self.learning_rate > 0, "training.learning_rate must be positive")
        _require(self.weight_decay >= 0, "training.weight_decay must be non-negative")
        _require(self.warmup_steps >= 0, "training.warmup_steps must be non-negative")
        _require(self.max_grad_norm > 0, "training.max_grad_norm must be positive")
        _require(self.patience >= 0, "training.patience must be non-negative")
        _require(self.seed >= 0, "training.seed must be non-negative")


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding strategy used to produce captions."""

    strategy: str = "beam"
    max_new_tokens: int = 30
    beam_size: int = 5
    length_penalty: float = 1.0
    top_p: float = 0.9
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate field values."""
        _require(
            self.strategy in DECODING_STRATEGIES,
            f"generation.strategy must be one of {DECODING_STRATEGIES}, got '{self.strategy}'",
        )
        _require(self.max_new_tokens > 0, "generation.max_new_tokens must be positive")
        _require(self.beam_size > 0, "generation.beam_size must be positive")
        _require(self.length_penalty >= 0, "generation.length_penalty must be non-negative")
        _require(0.0 < self.top_p <= 1.0, "generation.top_p must be in (0, 1]")
        _require(self.temperature > 0, "generation.temperature must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete description of one experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a configuration from a nested mapping, rejecting unknown keys."""
        sections = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - set(sections))
        _require(not unknown, f"unknown configuration section(s): {unknown}")
        section_types: dict[str, type[Any]] = {
            "data": DataConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "generation": GenerationConfig,
        }
        built = {
            name: _build_section(section_types[name], name, raw.get(name)) for name in sections
        }
        return cls(**built)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a configuration from a YAML file."""
        path = Path(path)
        _require(path.is_file(), f"configuration file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if raw is None:
            raw = {}
        _require(isinstance(raw, dict), f"{path} must contain a mapping at top level")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-serialisable nested dictionary."""
        plain: dict[str, Any] = _plain(dataclasses.asdict(self))
        return plain

    def to_yaml(self, path: str | Path) -> None:
        """Write the configuration to a YAML file, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


_T = TypeVar("_T")

_NUMERIC = (int, float)


def _build_section(section_type: type[_T], name: str, raw: Any) -> _T:
    if raw is None:
        raw = {}
    _require(isinstance(raw, dict), f"section '{name}' must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(section_type)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - set(fields))
    _require(not unknown, f"unknown key(s) in section '{name}': {unknown}")
    for key, value in raw.items():
        _check_type(f"{name}.{key}", value, str(fields[key].type))
    return section_type(**raw)


def _check_type(qualified_name: str, value: Any, annotation: str) -> None:
    """Reject values whose YAML type does not match the field annotation."""
    is_bool = isinstance(value, bool)
    if annotation == "bool":
        ok = is_bool
    elif annotation == "int":
        ok = isinstance(value, int) and not is_bool
    elif annotation == "float":
        ok = isinstance(value, _NUMERIC) and not is_bool
    else:  # "str" and "Path" are both written as strings in YAML
        ok = isinstance(value, str)
    _require(ok, f"{qualified_name} must be of type {annotation}, got {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value
