"""Shared test fixtures and integration gating.

Tests split into two tiers:

* **unit** (the default): pure Python logic — block allocator, prefix cache,
  scheduler, n-gram speculation, server request/response schema against a
  mocked engine. No model download, no GPU. This is what CI runs
  (``pytest -m "not integration"``).

* **integration** (``@pytest.mark.integration``): needs the real model
  weights on disk and, in practice, a CUDA device. Skipped automatically
  when the snapshot is absent so ``pytest`` never errors out on a fresh
  checkout.

The model directory comes from ``shrike.config.DEFAULT_MODEL_DIR`` (env var
``SHRIKE_MODEL_DIR``) — one source of truth instead of a string literal
copy-pasted across the suite.
"""

from __future__ import annotations

import os

import pytest

from shrike.config import DEFAULT_MODEL_DIR

MODEL_DIR = DEFAULT_MODEL_DIR


def model_available() -> bool:
    """True when a usable snapshot (a config.json) is present on disk."""
    return os.path.isfile(os.path.join(MODEL_DIR, "config.json"))


@pytest.fixture(scope="session")
def model_dir() -> str:
    if not model_available():
        pytest.skip(f"model snapshot not found at {MODEL_DIR!r}")
    return MODEL_DIR
