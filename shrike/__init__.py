"""shrike: an LLM inference engine built from scratch in PyTorch.

Paged KV cache, continuous batching, chunked prefill, prefix caching,
prompt-lookup speculative decoding, and an OpenAI-compatible server.

``SamplingParams`` is re-exported here because it is lightweight (no torch
import). ``LLMEngine`` lives in ``shrike.engine.engine`` and is imported
explicitly by callers, since importing it pulls in torch and transformers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from shrike.engine.request import SamplingParams

try:
    __version__ = version("shrike")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["SamplingParams", "__version__"]
