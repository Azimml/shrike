#!/usr/bin/env python3
"""Hit shrike's OpenAI-compatible chat-completions endpoint.

shrike serves the OpenAI wire format, so the official ``openai`` client works
unchanged — but this example uses only the Python standard library so it runs
with zero extra dependencies against a live ``shrike-serve`` process.

Start the server first:

    shrike-serve --model models_cache/qwen2.5-0.5b-instruct

Then, in another shell:

    python examples/openai_client.py "Explain paged attention in one sentence."
    python examples/openai_client.py --stream "Count to five."
    python examples/openai_client.py --base-url http://localhost:8000 "Hi"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def chat(base_url: str, prompt: str, *, max_tokens: int, temperature: float) -> str:
    """Non-streaming request: return the full assistant message."""
    payload = {
        "model": "shrike",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def chat_stream(base_url: str, prompt: str, *, max_tokens: int, temperature: float) -> None:
    """Streaming request: print Server-Sent Event deltas as they arrive."""
    payload = {
        "model": "shrike",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"]
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="user message to send")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true", help="stream tokens via SSE")
    args = parser.parse_args()

    try:
        if args.stream:
            chat_stream(
                args.base_url,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            reply = chat(
                args.base_url,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(reply)
    except urllib.error.URLError as err:
        print(
            f"could not reach shrike at {args.base_url}: {err.reason}\nis `shrike-serve` running?",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
