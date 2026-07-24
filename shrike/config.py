"""Centralized configuration constants for shrike.

Historically the model snapshot path was hard-coded as a string literal in
five different places (cli, server, benchmarks, tests). That made the default
model impossible to change without a find-and-replace, and coupled the test
suite to one specific on-disk layout. Everything now imports from here.

The default model directory can be overridden with the ``SHRIKE_MODEL_DIR``
environment variable, which is what lets CI and local dev point at whatever
snapshot they have downloaded without editing source.
"""

from __future__ import annotations

import os

#: Registered HF repo id -> the model family the loader should build.
#: The loader is Qwen2-architecture only today, but the registry keeps the
#: "which snapshot" decision out of call sites and documents the sizes the
#: engine has actually been run against.
MODEL_REGISTRY: dict[str, str] = {
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2",
}

#: Default on-disk snapshot used by the CLI, server, and benchmarks.
#: Overridable via the SHRIKE_MODEL_DIR env var.
DEFAULT_MODEL_DIR: str = os.environ.get("SHRIKE_MODEL_DIR", "models_cache/qwen2.5-0.5b-instruct")
