"""The experiment files shipped in ``configs/`` and the launch script."""

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from clipcap.config import ExperimentConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = sorted((ROOT / "configs").glob("*.yaml"))
SCRIPT = ROOT / "scripts" / "run_experiment.sh"


def test_expected_experiments_are_shipped() -> None:
    assert [path.stem for path in CONFIG_FILES] == [
        "mlp_finetune",
        "mlp_frozen",
        "transformer_frozen",
    ]


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda path: path.stem)
def test_experiment_file_is_valid_and_self_consistent(path: Path) -> None:
    config = ExperimentConfig.from_yaml(path)
    assert config.training.output_dir == Path("outputs") / path.stem
    assert config.model.mapper_type in path.stem
    assert config.model.freeze_gpt2 is ("frozen" in path.stem)
    # The prefix and the longest caption must fit in GPT-2's 1024 positions,
    # and so must the prefix and the longest generated caption.
    assert config.model.prefix_length + config.data.max_caption_tokens <= 1024
    assert config.model.prefix_length + config.generation.max_new_tokens <= 1024


def test_experiments_share_data_and_differ_only_where_intended() -> None:
    configs = [ExperimentConfig.from_yaml(path) for path in CONFIG_FILES]
    assert len({config.training.output_dir for config in configs}) == len(configs)
    # Comparable results need identical data, seed, batch size and decoding.
    assert len({config.data for config in configs}) == 1
    assert len({config.generation for config in configs}) == 1
    assert len({(c.training.seed, c.training.batch_size) for c in configs}) == 1
    assert (
        len({(c.model.clip_name, c.model.gpt2_name, c.model.prefix_length) for c in configs}) == 1
    )


@pytest.mark.parametrize("prefix_length", [5, 20, 40])
def test_prefix_ablations_only_change_the_prefix_length(prefix_length: int) -> None:
    baseline = ExperimentConfig.from_yaml(ROOT / "configs" / "mlp_frozen.yaml")
    ablation = ExperimentConfig.from_yaml(
        ROOT / "configs" / "ablations" / f"prefix_{prefix_length}.yaml"
    )
    assert ablation.model == dataclasses.replace(baseline.model, prefix_length=prefix_length)
    assert ablation.training == dataclasses.replace(
        baseline.training, output_dir=Path(f"outputs/ablations/prefix_{prefix_length}")
    )
    assert ablation.data == baseline.data
    assert ablation.generation == baseline.generation


def test_launch_script_is_executable_and_syntactically_valid() -> None:
    assert SCRIPT.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize(
    ("arguments", "code", "message"),
    [
        ([], 64, "usage:"),
        (["a.yaml", "--bogus"], 64, "usage:"),
        (["a.yaml", "--resume", "extra"], 64, "usage:"),
        (["does-not-exist.yaml"], 66, "configuration file not found"),
    ],
)
def test_launch_script_rejects_bad_invocations(
    arguments: list[str], code: int, message: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), *arguments], capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == code
    assert message in result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_launch_script_passes_shellcheck() -> None:
    subprocess.run(["shellcheck", str(SCRIPT)], check=True)
