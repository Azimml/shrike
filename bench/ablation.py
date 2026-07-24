"""Feature ablation: the same offline workload with each technology toggled.

Rungs 1-3 of the ladder already ablate the KV cache and batching (they ARE
the engine without those). This measures what the remaining toggles
contribute on top:

  full              defaults (prefix caching on, 512-token chunked budget)
  no_prefix_cache   enable_prefix_caching=False
  no_chunked_prefill max_tokens_per_step=8192 (prompts prefill in one shot;
                    offline throughput barely moves — the feature exists for
                    inter-token latency under load, see the serving A/B)
  spec_ngram        spec_ngram=2 (honest: general chat output is not
                    repetitive, so expect little gain on THIS workload;
                    see README for where it wins)

Usage: python -m bench.ablation [--num-prompts 64]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bench.prompts import PROMPTS
from bench.workload import target_lengths
from shrike.config import DEFAULT_MODEL_DIR
from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams

MODEL_DIR = DEFAULT_MODEL_DIR
RESULTS_PATH = Path("bench/results/ablation.json")

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_prefix_cache": {"enable_prefix_caching": False},
    "no_chunked_prefill": {"max_tokens_per_step": 8192},
    "spec_ngram": {"spec_ngram": 2},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prompts", type=int, default=64)
    args = parser.parse_args()

    prompts = PROMPTS[: args.num_prompts]
    lengths = target_lengths(len(PROMPTS))[: args.num_prompts]
    sampling = [SamplingParams(max_new_tokens=n, temperature=0.0, ignore_eos=True) for n in lengths]

    results: dict[str, dict] = {}
    for name, kwargs in VARIANTS.items():
        engine = LLMEngine(MODEL_DIR, **kwargs)
        started_at = time.perf_counter()
        outputs = engine.generate(prompts, sampling)
        seconds = time.perf_counter() - started_at
        tokens = sum(len(o) for o in outputs)
        metrics = engine.metrics()
        results[name] = {
            "tokens": tokens,
            "seconds": round(seconds, 3),
            "tok_s": round(tokens / seconds, 1),
            "steps": metrics["steps"],
            "prefix_cache_hit_rate": metrics["prefix_cache_hit_rate"],
            "spec_acceptance_rate": metrics["spec_acceptance_rate"],
        }
        print(
            f"{name:<20} {tokens} tok in {seconds:6.2f}s = {tokens / seconds:7.1f} tok/s "
            f"(steps={metrics['steps']}, cache_hit={metrics['prefix_cache_hit_rate']}, "
            f"spec_accept={metrics['spec_acceptance_rate']})"
        )
        del engine
        torch.cuda.empty_cache()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
