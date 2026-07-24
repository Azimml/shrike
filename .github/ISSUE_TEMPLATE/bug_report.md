---
name: Bug report
about: Report incorrect output, a crash, or a regression
title: "[bug] "
labels: bug
---

## What happened

A clear description of the bug and what you expected instead.

## Reproduction

Exact command(s) and, if relevant, the prompt or request payload:

```bash
# e.g. shrike-serve --model ... / curl ... / pytest -k ...
```

## Environment

- shrike version / commit:
- Python version:
- OS:
- PyTorch version and device (CPU / CUDA + GPU model):
- Attention backend (`einsum` / `triton`):
- Model (default is Qwen2.5-0.5B-Instruct):

## Logs / traceback

```
paste any error output here
```

## Notes

Does it reproduce on the einsum backend (CPU-capable) as well as Triton? That
helps localize whether the issue is in the kernel or the engine logic.
