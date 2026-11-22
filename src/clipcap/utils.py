"""Reproducibility, device selection and logging helpers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "clipcap"
_HANDLER_NAME = "clipcap-console"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) random number generators."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the device to run on.

    ``"auto"`` picks CUDA when available and falls back to CPU. An explicit
    CUDA request on a machine without CUDA raises instead of silently
    running on CPU.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device '{requested}' requested but CUDA is not available")
    return device


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure the package logger once and return it.

    Calling it again only updates the level, so handlers are never duplicated.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger named after ``name``."""
    suffix = name.removeprefix(f"{_ROOT_LOGGER_NAME}.")
    if suffix == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{suffix}")
