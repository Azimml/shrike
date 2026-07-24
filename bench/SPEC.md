# Benchmark harness spec (bench/)

Target hardware: single RTX 3050 Laptop 4GB. Model: Qwen2.5-0.5B-Instruct at
`models_cache/qwen2.5-0.5b-instruct`. Python at `.venv/bin/python`, run from repo root.

## Fixed workload (same everywhere)
- `bench/prompts.py`: `PROMPTS: list[str]` — 64 varied instruction prompts, 15–60 words each.
- Greedy (temperature=0), exactly 64 new tokens per request (force length: shrike
  `ignore_eos=True`; HF `min_new_tokens=64, max_new_tokens=64`).

## bench/baselines.py — the throughput ladder, rungs 1–3 (HF transformers, bf16, cuda)
CLI: `python -m bench.baselines --rung {1,2,3} [--num-prompts 16]`
- Rung 1 "no_kv_cache": bs=1, manual greedy loop; each step calls `model(full_ids).logits`
  (recompute whole prefix, `use_cache=False`). Use `--num-prompts 8` default here (it's slow) and
  extrapolate nothing — report measured tok/s only.
- Rung 2 "kv_cache": bs=1, manual greedy loop with `past_key_values` (`use_cache=True`).
- Rung 3 "static_batching": one `model.generate()` call over the whole prompt set left-padded
  as a single batch (or largest batch that fits 4GB; catch OOM and halve). Batch waits for the
  longest sequence — that's the point of the rung.
- Measure: wall seconds for the generation phase, generated tokens, tok/s.
- Append/merge results into `bench/results/results.json` as
  `{"rung_name": {"tokens": int, "seconds": float, "tok_s": float, "num_prompts": int}}`.
- Load HF model once per invocation; `torch.cuda.synchronize()` around timers.

## bench/bench_engine.py — rung 4 (shrike offline)
CLI: `python -m bench.bench_engine [--num-prompts 64] [--no-prefix-caching]`
```python
from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams

engine = LLMEngine("models_cache/qwen2.5-0.5b-instruct")  # kwargs: enable_prefix_caching: bool
outs = engine.generate(PROMPTS, SamplingParams(max_new_tokens=64, temperature=0.0, ignore_eos=True))
```
Time the `generate` call; write rung `"shrike_engine"` into the same results.json. Also print
`engine.metrics()` and store it under `"shrike_metrics"`.

## bench/load_gen.py — async load generator vs the HTTP server
Server (started separately): `python -m shrike.server.api` on 127.0.0.1:8000.
- POST /v1/completions JSON `{"prompt": str, "max_tokens": 64, "temperature": 0.0,
  "stream": true, "ignore_eos": true}` → SSE stream, lines `data: {"text": "..."}`,
  then `data: {"finish_reason": ...}`, then `data: [DONE]`.
CLI: `python -m bench.load_gen --concurrency 512 --num-requests 512 [--url http://127.0.0.1:8000]`
- asyncio + httpx, burst-open all requests, semaphore caps in-flight at --concurrency.
- Per request: TTFT (send → first SSE data), inter-token gaps, e2e latency, token count.
- Report + write `bench/results/load_test.json`: total wall s, aggregate tok/s, request
  throughput (req/s), TTFT p50/p99, inter-token latency p50/p99, e2e p50/p99, failures.
- Prompts: cycle PROMPTS. Exit nonzero if any request fails.

## bench/plots.py
CLI: `python -m bench.plots` — reads both result files, writes:
- `bench/results/throughput_ladder.png`: horizontal bar chart, tok/s per rung (log x-axis),
  value labels on bars, title includes model + GPU.
- `bench/results/latency_cdf.png`: CDF of inter-token latencies + TTFT from load_test.json.
matplotlib only, no seaborn, default style, tight_layout, dpi=150.

## Style
Plain Python, type hints, no new dependencies beyond torch/transformers/httpx/matplotlib/numpy.
Small files, no over-abstraction. Handle missing results.json entries gracefully in plots.
