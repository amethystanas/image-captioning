import dataclasses
from pathlib import Path

import pytest

from clipcap.config import ConfigError, ExperimentConfig, ModelConfig


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_valid() -> None:
    config = ExperimentConfig()
    assert config.model.mapper_type == "mlp"
    assert config.model.freeze_gpt2 is True
    assert config.generation.strategy == "beam"


def test_from_yaml_overrides_only_given_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        model:
          mapper_type: transformer
          prefix_length: 20
        training:
          learning_rate: 1.0e-4
          output_dir: outputs/run1
        """,
    )
    config = ExperimentConfig.from_yaml(path)
    assert config.model.mapper_type == "transformer"
    assert config.model.prefix_length == 20
    assert config.model.gpt2_name == "gpt2"
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert config.training.output_dir == Path("outputs/run1")
    assert config.data == ExperimentConfig().data


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    assert ExperimentConfig.from_yaml(_write(tmp_path, "")) == ExperimentConfig()


def test_yaml_round_trip(tmp_path: Path) -> None:
    original = ExperimentConfig(model=ModelConfig(mapper_type="transformer", prefix_length=7))
    path = tmp_path / "nested" / "config.yaml"
    original.to_yaml(path)
    assert ExperimentConfig.from_yaml(path) == original


def test_config_is_immutable() -> None:
    config = ExperimentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model.prefix_length = 3  # type: ignore[misc]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        ExperimentConfig.from_yaml(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        ExperimentConfig.from_yaml(_write(tmp_path, "model: [unclosed"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("optimizer: {}", "unknown configuration section"),
        ("model: {prefix_len: 10}", "unknown key"),
        ("model: [1, 2]", "must be a mapping"),
        ("- 1\n- 2", "mapping at top level"),
        ("model: {mapper_type: cnn}", "mapper_type"),
        ("model: {prefix_length: 0}", "prefix_length"),
        ("model: {prefix_length: '10'}", "must be of type int"),
        ("model: {prefix_length: true}", "must be of type int"),
        ("model: {freeze_gpt2: 1}", "must be of type bool"),
        ("model: {mapper_dropout: 1.0}", "mapper_dropout"),
        ("training: {learning_rate: 0}", "learning_rate"),
        ("training: {learning_rate: fast}", "must be of type float"),
        ("training: {epochs: -1}", "epochs"),
        ("training: {seed: -3}", "seed"),
        ("data: {root: 12}", "must be of type Path"),
        ("data: {max_caption_tokens: 0}", "max_caption_tokens"),
        ("generation: {strategy: random}", "strategy"),
        ("generation: {top_p: 0.0}", "top_p"),
        ("generation: {length_penalty: -1}", "length_penalty"),
        ("generation: {temperature: 0}", "temperature"),
    ],
)
def test_invalid_configurations_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        ExperimentConfig.from_yaml(_write(tmp_path, text))
