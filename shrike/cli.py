"""Interactive terminal chat for the in-process shrike engine."""

from __future__ import annotations

import argparse
import readline
import sys
import time
from typing import cast

from shrike.config import DEFAULT_MODEL_DIR
from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a local shrike model")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--spec-ngram", type=int, default=0)
    return parser.parse_args()


def _ansi(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_help() -> None:
    print(
        """Commands:
  /help                Show this help
  /clear               Clear conversation history
  /metrics             Show engine metrics (cache hits, speculation, blocks)
  /reset               Zero the metrics counters (clean measurements)
  /temp <float>        Set temperature; 0 uses greedy decoding
  /max <int>           Set maximum new tokens
  /spec on [ngram] [k] Enable prompt-lookup speculation (default ngram=2 k=4)
  /spec off            Disable speculation
  /cache on|off        Toggle prefix caching live
  /budget <int>        Set scheduler token budget per step (chunked prefill)
  /bench <n> [tokens]  Fire n concurrent requests (default 64 tokens each)
                       and report aggregate throughput — watch batching scale
  /system <text>       Set or replace the system message
  /exit                Quit

Experiments to try:
  /bench 1  then  /bench 8  then  /bench 64   -> continuous batching scaling
  /cache off, /bench 32, /reset, /cache on, /bench 32, /metrics -> prefix cache
  /temp 0, /spec on, then: Repeat exactly five times: <sentence>  -> speculation
  /budget 64 vs /budget 512 -> chunked prefill (TTFT in the status line)"""
    )


def _run_bench(engine: LLMEngine, count: int, tokens: int) -> None:
    prompt = "Explain why paged memory management helps LLM serving."
    sampling = SamplingParams(max_new_tokens=tokens, temperature=0.0, ignore_eos=True)
    try:
        for _ in range(count):
            engine.add_request(prompt, sampling)
    except ValueError as error:
        print(f"[error] {error}")
        return
    started_at = time.perf_counter()
    generated = 0
    steps = 0
    while engine.has_work:
        generated += len(engine.step())
        steps += 1
    elapsed = time.perf_counter() - started_at
    metrics = engine.metrics()
    print(
        f"{count} reqs x {tokens} tok -> {generated} tok in {elapsed:.2f}s "
        f"= {generated / elapsed:.0f} tok/s aggregate "
        f"({generated / elapsed / count:.1f} tok/s per request, {steps} steps, "
        f"cache_hit={metrics['prefix_cache_hit_rate']})"
    )


def _print_metrics(engine: LLMEngine) -> None:
    metrics = engine.metrics()
    width = max(len(key) for key in metrics)
    for key, value in metrics.items():
        print(f"{key:<{width}}  {value}")


def _stream_reply(
    engine: LLMEngine,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    use_color: bool,
) -> str | None:
    request_id: int | None = None
    generated: list[int] = []
    decode_buffer: list[int] = []
    reply_parts: list[str] = []
    started_at = time.perf_counter()
    first_token_at: float | None = None

    try:
        encoded = cast(
            "dict[str, list[int]]",
            engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True),
        )
        token_ids: list[int] = list(encoded["input_ids"])
        sampling = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        try:
            request_id = engine.add_request(token_ids, sampling, chat=False)
        except ValueError as error:
            print(f"[error] {error}")
            return None

        while engine.has_work:
            for output in engine.step():
                if output.req_id != request_id:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                generated.append(output.token_id)
                decode_buffer.append(output.token_id)
                text = cast("str", engine.tokenizer.decode(decode_buffer, skip_special_tokens=True))
                if not text.endswith("�"):
                    print(text, end="", flush=True)
                    reply_parts.append(text)
                    decode_buffer.clear()
    except KeyboardInterrupt:
        if request_id is not None:
            request = engine.requests.get(request_id)
            if request is not None:
                try:
                    engine.scheduler.finish(request, "aborted")
                except ValueError:
                    if request in engine.scheduler.waiting:
                        engine.scheduler.waiting.remove(request)
                    engine.block_manager.free(request)
                del engine.requests[request_id]
        print("\n[aborted]")
        return None

    elapsed = time.perf_counter() - started_at
    ttft_ms = (first_token_at - started_at) * 1000 if first_token_at is not None else 0.0
    rate = len(generated) / max(elapsed, 1e-9)
    print()
    status = f"⏱ {len(generated)} tok · {elapsed:.1f}s · {rate:.0f} tok/s · TTFT {ttft_ms:.0f}ms"
    print(_ansi(status, "2", use_color))
    return "".join(reply_parts)


def main() -> None:
    args = _parse_args()
    readline.set_auto_history(True)
    use_color = sys.stdout.isatty()

    print(f"Loading {args.model} ...", flush=True)
    load_started = time.perf_counter()
    engine = LLMEngine(args.model, spec_ngram=args.spec_ngram)
    load_time = time.perf_counter() - load_started
    print(
        f"shrike ready · {args.model} · {engine.block_manager.num_blocks} KV blocks "
        f"· loaded in {load_time:.1f}s"
    )

    messages: list[dict[str, str]] = []
    temperature = 0.7
    max_new_tokens = 512
    prompt = "\001\033[1;36m\002shrike> \001\033[0m\002" if use_color else "shrike> "

    while True:
        try:
            line = input(prompt)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nUse /exit to quit.")
            continue

        if not line.strip():
            continue

        if line.startswith("/"):
            command, _, argument = line.strip().partition(" ")
            argument = argument.strip()

            if command == "/exit":
                break
            if command == "/help":
                _print_help()
            elif command == "/clear":
                messages.clear()
                print("Conversation cleared.")
            elif command == "/metrics":
                _print_metrics(engine)
            elif command == "/temp":
                try:
                    value = float(argument)
                    if not 0 <= value < float("inf"):
                        raise ValueError
                except ValueError:
                    print("Usage: /temp <non-negative float>")
                else:
                    temperature = value
                    print(f"Temperature set to {temperature:g}.")
            elif command == "/max":
                try:
                    value = int(argument)
                    if value <= 0:
                        raise ValueError
                except ValueError:
                    print("Usage: /max <positive int>")
                else:
                    max_new_tokens = value
                    print(f"Maximum new tokens set to {max_new_tokens}.")
            elif command == "/spec":
                parts = argument.split()
                if parts and parts[0] == "on":
                    try:
                        engine.scheduler.spec_ngram = int(parts[1]) if len(parts) > 1 else 2
                        engine.scheduler.spec_k = int(parts[2]) if len(parts) > 2 else 4
                    except ValueError:
                        print("Usage: /spec on [ngram] [k]")
                    else:
                        print(
                            f"Speculation on (ngram={engine.scheduler.spec_ngram}, "
                            f"k={engine.scheduler.spec_k}); takes effect at /temp 0."
                        )
                elif parts and parts[0] == "off":
                    engine.scheduler.spec_ngram = 0
                    print("Speculation off.")
                else:
                    print("Usage: /spec on [ngram] [k] | /spec off")
            elif command == "/cache":
                if argument in ("on", "off"):
                    engine.block_manager.enable_prefix_caching = argument == "on"
                    print(f"Prefix caching {argument}.")
                else:
                    print("Usage: /cache on|off")
            elif command == "/budget":
                try:
                    value = int(argument)
                    if value < 1:
                        raise ValueError
                except ValueError:
                    print("Usage: /budget <positive int>")
                else:
                    engine.scheduler.max_tokens_per_step = value
                    print(f"Scheduler token budget set to {value} per step.")
            elif command == "/reset":
                from shrike.engine.engine import EngineStats

                engine.stats = EngineStats()
                engine.block_manager.cache_hit_blocks = 0
                engine.block_manager.cache_query_blocks = 0
                engine.scheduler.num_preemptions = 0
                print("Metrics counters reset.")
            elif command == "/bench":
                parts = argument.split()
                try:
                    count = int(parts[0])
                    tokens = int(parts[1]) if len(parts) > 1 else 64
                    if count < 1 or tokens < 1:
                        raise ValueError
                except (IndexError, ValueError):
                    print("Usage: /bench <num_requests> [tokens_each]")
                else:
                    _run_bench(engine, count, tokens)
            elif command == "/system":
                if not argument:
                    print("Usage: /system <text>")
                elif messages and messages[0]["role"] == "system":
                    messages[0]["content"] = argument
                    print("System message updated.")
                else:
                    messages.insert(0, {"role": "system", "content": argument})
                    print("System message set.")
            else:
                print(f"Unknown command: {command}. Type /help for commands.")
            continue

        messages.append({"role": "user", "content": line})
        reply = _stream_reply(engine, messages, temperature, max_new_tokens, use_color)
        if reply is not None:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
