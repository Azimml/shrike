"""Offline throughput benchmark for the Shrike engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from bench.prompts import PROMPTS
from bench.workload import target_lengths
from shrike.config import DEFAULT_MODEL_DIR
from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams

MODEL_DIR = DEFAULT_MODEL_DIR
RESULTS_PATH = Path("bench/results/results.json")


def _write_results(result: dict[str, int | float], metrics: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as result_file:
            results = json.load(result_file)
        if not isinstance(results, dict):
            raise ValueError(f"Expected a JSON object in {RESULTS_PATH}")
    else:
        results = {}
    results["shrike_engine"] = result
    results["shrike_metrics"] = metrics
    with RESULTS_PATH.open("w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
        result_file.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--no-prefix-caching", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_prompts <= len(PROMPTS):
        parser.error(f"--num-prompts must be between 1 and {len(PROMPTS)}")
    return args


def main() -> None:
    args = _parse_args()
    engine = LLMEngine(MODEL_DIR, enable_prefix_caching=not args.no_prefix_caching)
    prompts = PROMPTS[: args.num_prompts]
    lengths = target_lengths(len(PROMPTS))[: args.num_prompts]
    sampling = [SamplingParams(max_new_tokens=n, temperature=0.0, ignore_eos=True) for n in lengths]

    started_at = time.perf_counter()
    outputs = engine.generate(prompts, sampling)
    seconds = time.perf_counter() - started_at
    tokens = sum(len(output) for output in outputs)
    result: dict[str, int | float] = {
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds,
        "num_prompts": len(prompts),
    }
    metrics = engine.metrics()
    _write_results(result, metrics)

    print(f"shrike_engine: {tokens} tokens in {seconds:.3f}s ({tokens / seconds:.2f} tok/s)")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
