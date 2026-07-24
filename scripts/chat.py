"""Quick smoke test: stream one completion through the offline engine.

Usage: .venv/bin/python scripts/chat.py "your prompt" [--max-tokens 128]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from shrike.config import DEFAULT_MODEL_DIR
from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Explain how a paged KV cache works, briefly.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    t0 = time.perf_counter()
    engine = LLMEngine(args.model)
    print(
        f"[engine up in {time.perf_counter() - t0:.1f}s, "
        f"{engine.block_manager.num_blocks} KV blocks]",
        file=sys.stderr,
    )

    engine.add_request(args.prompt, SamplingParams(args.max_tokens, args.temperature))
    buf: list[int] = []
    t0, n = time.perf_counter(), 0
    while engine.has_work:
        for out in engine.step():
            n += 1
            buf.append(out.token_id)
            text = engine.tokenizer.decode(buf, skip_special_tokens=True)
            if not text.endswith("�"):
                print(text, end="", flush=True)
                buf.clear()
    dt = time.perf_counter() - t0
    print(f"\n[{n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
