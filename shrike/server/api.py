"""FastAPI serving layer.

Endpoints:
  POST /v1/completions        legacy text-completion (SSE or JSON)
  POST /v1/chat/completions   OpenAI-compatible chat completions (SSE or JSON)
  GET  /v1/models             OpenAI-compatible model listing
  GET  /health                liveness
  GET  /metrics               engine counters

The chat-completions endpoint speaks the OpenAI wire format, so any OpenAI
client library (openai-python, LangChain, etc.) can point its base_url at
this server unchanged. The request/response schema and SSE framing are exact
enough to unit-test against a mocked engine with no GPU or model download.

Run:  python -m shrike.server.api --model models_cache/qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from shrike.config import DEFAULT_MODEL_DIR
from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams
from shrike.server.async_engine import AsyncEngine

_engine_config: dict = {}
async_engine: AsyncEngine | None = None

MODEL_NAME = "shrike"


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str = MODEL_NAME
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    ignore_eos: bool = False


def _sampling_from(
    max_tokens: int, temperature: float, top_p: float, ignore_eos: bool
) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        ignore_eos=ignore_eos,
    )


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global async_engine
    async_engine = AsyncEngine(LLMEngine(**_engine_config))
    async_engine.start()
    yield
    await async_engine.stop()


app = FastAPI(title="shrike", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict:
    assert async_engine is not None
    return async_engine.engine.metrics()


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compatible model listing (clients probe this on startup)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "shrike",
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Legacy text completions
# --------------------------------------------------------------------------- #
@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    assert async_engine is not None
    params = _sampling_from(req.max_tokens, req.temperature, req.top_p, req.ignore_eos)
    try:
        req_id, queue = await async_engine.submit(req.prompt, params)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    if req.stream:

        async def sse() -> AsyncIterator[str]:
            while True:
                ev = await queue.get()
                if ev.text:
                    yield f"data: {json.dumps({'text': ev.text})}\n\n"
                if ev.finished:
                    yield f"data: {json.dumps({'finish_reason': ev.finish_reason})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(sse(), media_type="text/event-stream")

    chunks: list[str] = []
    finish_reason = None
    num_tokens = 0
    while True:
        ev = await queue.get()
        num_tokens += 1
        chunks.append(ev.text)
        if ev.finished:
            finish_reason = ev.finish_reason
            break
    return JSONResponse(
        {
            "id": req_id,
            "text": "".join(chunks),
            "finish_reason": finish_reason,
            "num_tokens": num_tokens,
        }
    )


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat completions
# --------------------------------------------------------------------------- #
def _openai_finish_reason(reason: str | None) -> str:
    """Map shrike's finish reasons onto the OpenAI vocabulary."""
    return "stop" if reason == "stop" else "length"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    assert async_engine is not None
    params = _sampling_from(req.max_tokens, req.temperature, req.top_p, req.ignore_eos)
    messages = [m.model_dump() for m in req.messages]
    prompt_ids = async_engine.render_chat(messages)

    try:
        req_id, queue = await async_engine.submit(prompt_ids, params, chat=False)
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=422,
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if req.stream:

        async def sse() -> AsyncIterator[str]:
            # first chunk carries the assistant role, per the OpenAI protocol
            first = _chat_chunk(completion_id, created, req.model, delta={"role": "assistant"})
            yield f"data: {json.dumps(first)}\n\n"
            while True:
                ev = await queue.get()
                if ev.text:
                    chunk = _chat_chunk(
                        completion_id, created, req.model, delta={"content": ev.text}
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                if ev.finished:
                    final = _chat_chunk(
                        completion_id,
                        created,
                        req.model,
                        delta={},
                        finish_reason=_openai_finish_reason(ev.finish_reason),
                    )
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts: list[str] = []
    finish_reason: str | None = None
    completion_tokens = 0
    while True:
        ev = await queue.get()
        completion_tokens += 1
        parts.append(ev.text)
        if ev.finished:
            finish_reason = ev.finish_reason
            break

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": _openai_finish_reason(finish_reason),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_ids) + completion_tokens,
            },
        }
    )


def _chat_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens-per-step", type=int, default=512)
    parser.add_argument("--max-running", type=int, default=256)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--attention-backend", choices=["einsum", "triton"], default="einsum")
    args = parser.parse_args()
    _engine_config.update(
        model_dir=args.model,
        max_tokens_per_step=args.max_tokens_per_step,
        max_running=args.max_running,
        enable_prefix_caching=not args.no_prefix_caching,
        attention_backend=args.attention_backend,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
