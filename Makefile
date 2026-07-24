# Developer shortcuts. These mirror what CI runs (.github/workflows/ci.yml) so
# `make check` locally is the same gate a pull request must pass.
#
# PY lets you point at a specific interpreter, e.g. `make test PY=.venv/bin/python`.
PY ?= python

.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test test-all check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with dev + bench + triton extras
	$(PY) -m pip install -e ".[dev,bench,triton]"

fmt:  ## Auto-format with ruff
	$(PY) -m ruff format .

lint:  ## Lint (format check + ruff check), matching CI
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .

typecheck:  ## Run mypy (advisory, as in CI)
	$(PY) -m mypy shrike/ || true

test:  ## Fast unit tests: no model download, no GPU
	CUDA_VISIBLE_DEVICES="" $(PY) -m pytest -q -m "not integration"

test-all:  ## Full suite including integration (needs model weights + GPU)
	$(PY) -m pytest -q

check: lint test  ## The pre-push gate: lint + unit tests

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
