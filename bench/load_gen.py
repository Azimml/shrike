"""Asynchronous burst load generator for the Shrike HTTP server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bench.prompts import PROMPTS

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "load_test.json"
NEW_TOKENS = 64
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class RequestMetrics:
    request_index: int
    ttft_ms: float
    inter_token_ms: list[float]
    e2e_ms: float
    token_count: int
    sse_text_chunks: int

    def as_dict(self) -> dict[str, int | float | list[float]]:
        return {
            "request_index": self.request_index,
            "ttft_ms": self.ttft_ms,
            "inter_token_ms": self.inter_token_ms,
            "e2e_ms": self.e2e_ms,
            "token_count": self.token_count,
            "sse_text_chunks": self.sse_text_chunks,
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


async def _measure_request(
    client: httpx.AsyncClient,
    endpoint: str,
    semaphore: asyncio.Semaphore,
    request_index: int,
    prompt: str,
    delay_s: float = 0.0,
    api: str = "shrike",
    model: str = DEFAULT_MODEL,
) -> RequestMetrics:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    payload = {
        "prompt": prompt,
        "max_tokens": NEW_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    if api == "openai":
        payload = {"model": model, **payload}

    async with semaphore:
        started_at = time.perf_counter()
        text_event_times: list[float] = []
        finish_reason: str | None = None
        saw_finish = False
        saw_done = False
        completed_at = started_at

        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if data == "[DONE]":
                    completed_at = time.perf_counter()
                    saw_done = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid SSE JSON: {data!r}") from error
                if not isinstance(event, dict):
                    raise RuntimeError("SSE data must decode to a JSON object")
                if api == "openai":
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        raise RuntimeError("OpenAI SSE data must include a choice")
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        raise RuntimeError("OpenAI SSE choice must be a JSON object")
                    text = choice.get("text")
                    if not isinstance(text, str):
                        raise RuntimeError("OpenAI SSE text value must be a string")
                    if text:
                        text_event_times.append(time.perf_counter())
                    if "finish_reason" in choice:
                        reason = choice["finish_reason"]
                        if reason is not None and not isinstance(reason, str):
                            raise RuntimeError("OpenAI SSE finish_reason must be a string or null")
                        if reason is not None:
                            finish_reason = reason
                            saw_finish = True
                elif "text" in event:
                    if not isinstance(event["text"], str):
                        raise RuntimeError("SSE text value must be a string")
                    text_event_times.append(time.perf_counter())
                elif "finish_reason" in event:
                    reason = event["finish_reason"]
                    if reason is not None and not isinstance(reason, str):
                        raise RuntimeError("SSE finish_reason must be a string or null")
                    finish_reason = reason
                    saw_finish = True
                else:
                    raise RuntimeError(f"unexpected SSE event: {event!r}")

        if not saw_done:
            raise RuntimeError("SSE stream ended before [DONE]")
        if not saw_finish:
            raise RuntimeError("SSE stream did not include a finish_reason")
        if finish_reason != "length":
            raise RuntimeError(f"expected finish_reason 'length', got {finish_reason!r}")
        if not text_event_times:
            raise RuntimeError("SSE stream did not include any text data")

        inter_token_ms = [
            (current - previous) * 1000.0
            for previous, current in zip(text_event_times, text_event_times[1:], strict=False)
        ]
        return RequestMetrics(
            request_index=request_index,
            ttft_ms=(text_event_times[0] - started_at) * 1000.0,
            inter_token_ms=inter_token_ms,
            e2e_ms=(completed_at - started_at) * 1000.0,
            token_count=NEW_TOKENS,
            sse_text_chunks=len(text_event_times),
        )


async def _run_load_test(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    endpoint = f"{args.url.rstrip('/')}/v1/completions"
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )

    wall_started_at = time.perf_counter()
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                _measure_request(
                    client,
                    endpoint,
                    semaphore,
                    request_index,
                    _make_prompt(request_index, args.long_every, args.long_repeats),
                    (request_index / max(1, args.num_requests - 1)) * args.stagger_s,
                    args.api,
                    args.model,
                )
            )
            for request_index in range(args.num_requests)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    total_wall_s = time.perf_counter() - wall_started_at

    successful: list[RequestMetrics] = []
    failure_details: list[dict[str, int | str]] = []
    for request_index, outcome in enumerate(outcomes):
        if isinstance(outcome, BaseException):
            failure_details.append(
                {
                    "request_index": request_index,
                    "error": f"{type(outcome).__name__}: {outcome}",
                }
            )
        else:
            successful.append(outcome)

    ttft_ms = [metric.ttft_ms for metric in successful]
    inter_token_ms = [gap for metric in successful for gap in metric.inter_token_ms]
    e2e_ms = [metric.e2e_ms for metric in successful]
    total_tokens = sum(metric.token_count for metric in successful)
    successful_requests = len(successful)
    failures = len(failure_details)

    summary: dict[str, Any] = {
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "successful_requests": successful_requests,
        "total_tokens": total_tokens,
        "total_wall_s": total_wall_s,
        "aggregate_tok_s": total_tokens / total_wall_s,
        "request_throughput_req_s": successful_requests / total_wall_s,
        "ttft_p50_ms": _percentile(ttft_ms, 50),
        "ttft_p99_ms": _percentile(ttft_ms, 99),
        "inter_token_latency_p50_ms": _percentile(inter_token_ms, 50),
        "inter_token_latency_p99_ms": _percentile(inter_token_ms, 99),
        "e2e_p50_ms": _percentile(e2e_ms, 50),
        "e2e_p99_ms": _percentile(e2e_ms, 99),
        "failures": failures,
    }
    report = {
        **summary,
        "requests": [metric.as_dict() for metric in successful],
        "failure_details": failure_details,
    }

    results_path = Path(args.out) if args.out else RESULTS_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as result_file:
        json.dump(report, result_file, indent=2)
        result_file.write("\n")
    print(json.dumps(summary, indent=2))
    return report, failures


_LONG_FILLER = (
    "Background context, part of a long document under discussion: the paged "
    "key-value cache divides GPU memory into fixed-size blocks, the scheduler "
    "assigns blocks to sequences on demand, and freed blocks return to a pool "
    "where content hashes make them reusable by later requests. "
)


def _make_prompt(request_index: int, long_every: int, long_repeats: int) -> str:
    base = PROMPTS[request_index % len(PROMPTS)]
    if long_every and request_index % long_every == 0:
        return _LONG_FILLER * long_repeats + base
    return base


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=_positive_int, default=512)
    parser.add_argument("--num-requests", type=_positive_int, default=512)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api", choices=("shrike", "openai"), default="shrike")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--long-every",
        type=int,
        default=0,
        help="every Nth request gets a long prompt (0 = off); used to measure "
        "how chunked prefill protects inter-token latency",
    )
    parser.add_argument(
        "--long-repeats",
        type=int,
        default=40,
        help="filler paragraph repeats for long prompts (~40 tokens each)",
    )
    parser.add_argument("--out", default=None, help="results path override")
    parser.add_argument(
        "--stagger-s",
        type=float,
        default=0.0,
        help="spread request starts over this many seconds (0 = single burst); "
        "staggered long prompts show how chunked prefill protects in-flight "
        "decode latency",
    )
    return parser.parse_args()


def main() -> None:
    _, failures = asyncio.run(_run_load_test(_parse_args()))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
