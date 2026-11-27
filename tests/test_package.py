"""Smoke tests for the installed package."""

import re

import clipcap


def test_package_exposes_semantic_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", clipcap.__version__)
