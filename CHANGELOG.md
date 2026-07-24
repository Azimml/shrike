# Changelog

All notable changes to shrike are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CONTRIBUTING.md`, `Makefile`, `.editorconfig`, and GitHub issue/PR templates
  to standardize the contributor workflow.
- `examples/` directory with a runnable OpenAI-compatible client script.
- Additional CPU-only unit tests for the request lifecycle, configuration
  registry, block-manager boundaries, and the scheduler.

## [0.1.0]

Initial public release.

### Added
- Paged KV cache allocator (PagedAttention) with hash-based automatic prefix
  caching and LRU eviction.
- Iteration-level continuous batching scheduler with a Sarathi-Serve token
  budget, chunked prefill, and discard-and-recompute preemption.
- Prompt-lookup (n-gram) speculative decoding, verified in a single batched
  forward pass so outputs match greedy decoding exactly.
- Two attention backends behind one interface: pure-einsum SDPA (default,
  CPU-capable) and a Triton fused paged-decode kernel (Linux/CUDA).
- OpenAI-compatible FastAPI server (`/v1/chat/completions`, `/v1/completions`,
  `/v1/models`) with SSE streaming, plus `/health` and `/metrics`.
- Interactive terminal CLI with live metrics and prefix-cache/speculation
  toggles.
- Benchmark suite: throughput ladder, feature ablation, and a head-to-head
  against vLLM on identical hardware.
- Two-tier test suite (unit + integration) with HF-transformers parity checks,
  and CI running lint, advisory mypy, and the unit tier on Python 3.11/3.12.

[Unreleased]: https://github.com/Azimml/shrike/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azimml/shrike/releases/tag/v0.1.0
