"""Create throughput and latency plots from saved benchmark results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
THROUGHPUT_PATH = RESULTS_DIR / "throughput_ladder.png"
LATENCY_PATH = RESULTS_DIR / "latency_cdf.png"
RUNG_LABELS = [
    ("no_kv_cache", "No KV cache"),
    ("kv_cache", "KV cache"),
    ("static_batching", "Static batching"),
    ("shrike_engine", "Shrike engine"),
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as result_file:
            value = json.load(result_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite_nonnegative(values: list[Any]) -> np.ndarray:
    cleaned: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number) and number >= 0.0:
            cleaned.append(number)
    return np.asarray(cleaned, dtype=float)


def _plot_throughput(results: dict[str, Any]) -> None:
    labels: list[str] = []
    throughputs: list[float] = []
    for key, label in RUNG_LABELS:
        entry = results.get(key)
        if not isinstance(entry, dict):
            continue
        value = entry.get("tok_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        throughput = float(value)
        if not math.isfinite(throughput) or throughput <= 0.0:
            continue
        labels.append(label)
        throughputs.append(throughput)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    if throughputs:
        positions = np.arange(len(throughputs))
        bars = axis.barh(positions, throughputs)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xscale("log")
        axis.bar_label(
            bars,
            labels=[f"{value:.1f} tok/s" for value in throughputs],
            padding=4,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No throughput results available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    axis.set_xlabel("Generated tokens per second (log scale)")
    axis.set_title("Qwen2.5-0.5B-Instruct throughput — RTX 3050 Laptop 4GB")
    figure.tight_layout()
    figure.savefig(THROUGHPUT_PATH, dpi=150)
    plt.close(figure)


def _plot_cdf(axis: Any, samples: np.ndarray, label: str) -> None:
    ordered = np.sort(samples)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
    axis.plot(ordered, cumulative, label=label)


def _plot_latency(load_results: dict[str, Any]) -> None:
    request_results = load_results.get("requests", [])
    if not isinstance(request_results, list):
        request_results = []

    ttft_values: list[Any] = []
    inter_token_values: list[Any] = []
    for request in request_results:
        if not isinstance(request, dict):
            continue
        ttft_values.append(request.get("ttft_ms"))
        gaps = request.get("inter_token_ms", [])
        if isinstance(gaps, list):
            inter_token_values.extend(gaps)

    ttft = _finite_nonnegative(ttft_values)
    inter_token = _finite_nonnegative(inter_token_values)
    figure, axis = plt.subplots(figsize=(8, 5))
    if len(inter_token):
        _plot_cdf(axis, inter_token, "Inter-token latency")
    if len(ttft):
        _plot_cdf(axis, ttft, "TTFT")
    if len(inter_token) or len(ttft):
        axis.legend()
        axis.grid(True, alpha=0.3)
    else:
        axis.text(
            0.5,
            0.5,
            "No latency samples available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("Cumulative fraction")
    axis.set_title("Shrike HTTP load-test latency CDF")
    axis.set_ylim(0.0, 1.01)
    figure.tight_layout()
    figure.savefig(LATENCY_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_throughput(_load_json(RESULTS_DIR / "results.json"))
    _plot_latency(_load_json(RESULTS_DIR / "load_test.json"))
    print(f"Wrote {THROUGHPUT_PATH}")
    print(f"Wrote {LATENCY_PATH}")


if __name__ == "__main__":
    main()
