import logging
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from clipcap.utils import get_logger, resolve_device, seed_everything, setup_logging


@pytest.fixture
def clean_package_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("clipcap")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers.clear()
    yield logger
    logger.handlers[:], logger.level, logger.propagate = saved[0], saved[1], saved[2]


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_seed_everything_makes_draws_reproducible() -> None:
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seeds_give_different_draws() -> None:
    seed_everything(1)
    first = _draw()
    seed_everything(2)
    assert _draw() != first


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_resolve_device_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_setup_logging_is_idempotent(clean_package_logger: logging.Logger) -> None:
    setup_logging(logging.INFO)
    logger = setup_logging(logging.DEBUG)
    assert logger is clean_package_logger
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_get_logger_returns_package_children() -> None:
    assert get_logger("clipcap.data.dataset").name == "clipcap.data.dataset"
    assert get_logger("training").name == "clipcap.training"
    assert get_logger("clipcap").name == "clipcap"
