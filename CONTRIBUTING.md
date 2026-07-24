# Contributing to shrike

Thanks for your interest. shrike is a from-scratch LLM inference engine, so the
bar for changes is correctness first and clarity second — the code is meant to
be read as an explanation of *how* production inference works, not just to run.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # ruff, pytest, mypy
python scripts/download_model.py # optional: only needed for integration tests
```

Triton (`.[triton]`) and matplotlib (`.[bench]`) are optional extras; the
default einsum attention path and the unit tests need neither.

## The pre-push gate

Everything CI enforces is behind one target:

```bash
make check     # ruff format --check, ruff check, and the unit test tier
```

Run it before opening a pull request. Equivalently, by hand:

```bash
ruff format .                                    # auto-fix formatting
ruff check .                                     # lint (must pass)
pytest -m "not integration"                      # unit tier (no model, no GPU)
```

Six tests skip without a CUDA device (the Triton kernel numerics); that is
expected and fine.

## Test tiers

- **Unit** (`-m "not integration"`) — pure logic: the block allocator, prefix
  cache, scheduler, n-gram drafting, the sampler on synthetic logits, and the
  server schema against a mocked engine. These run anywhere and must stay
  green. New CPU-testable logic should ship with a unit test.
- **Integration** (`-m integration`) — parity against HF transformers and
  paged/batched correctness. These need the model weights on disk and, in
  practice, a GPU. Mark anything requiring a model or CUDA with
  `@pytest.mark.integration` so the default run stays fast and hermetic.

## Style

- `ruff` owns formatting and linting; config lives in `pyproject.toml`. Do not
  hand-format around it.
- `mypy` runs in advisory mode — it reports issues but does not block. New code
  is expected to be type-hinted; annotate what you touch.
- Keep commits small and self-contained, with a conventional-commit subject
  (`feat:`, `fix:`, `docs:`, `test:`, `perf:`, `refactor:`, `chore:`).

## Scope

Single GPU, single model, bf16. Tensor/pipeline parallelism, quantization, and
LoRA are out of scope by design — see the "Limitations" and "Future work"
sections of the README before proposing large features.
