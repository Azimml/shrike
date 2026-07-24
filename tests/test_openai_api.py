"""Wire-format tests for the OpenAI-compatible server endpoints.

These drive the real FastAPI app through Starlette's in-process TestClient
with a *mocked* engine, so they validate routing, request validation,
response schema, and SSE framing with no model download and no GPU.

The app is used WITHOUT the ``with TestClient(...)`` context-manager form on
purpose: entering the context runs the FastAPI lifespan, which would try to
construct a real LLMEngine. Instead we install a fake AsyncEngine on the
module and hit the routes directly.
"""

from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from shrike.server import api
from shrike.server.async_engine import TokenEvent


class FakeAsyncEngine:
    """Stands in for AsyncEngine: scripted token streams, no model."""

    def __init__(self, tokens: list[str], finish_reason: str = "stop"):
        self._tokens = tokens
        self._finish_reason = finish_reason
        self.last_prompt: str | list[int] | None = None
        self.last_chat: bool | None = None
        self.rendered_messages: list[dict] | None = None
        self.raise_on_submit: str | None = None

        class _Engine:
            def metrics(self_inner) -> dict:
                return {"steps": 0, "tokens_generated": 0}

        self.engine = _Engine()

    def render_chat(self, messages: list[dict]) -> list[int]:
        self.rendered_messages = messages
        # deterministic fake tokenization: one id per message, length-stable
        return list(range(1, len(messages) + 1))

    async def submit(self, prompt, sampling, chat: bool = True):
        self.last_prompt = prompt
        self.last_chat = chat
        if self.raise_on_submit is not None:
            raise ValueError(self.raise_on_submit)
        queue: asyncio.Queue[TokenEvent] = asyncio.Queue()
        for i, text in enumerate(self._tokens):
            last = i == len(self._tokens) - 1
            queue.put_nowait(
                TokenEvent(
                    token_id=i,
                    text=text,
                    finished=last,
                    finish_reason=self._finish_reason if last else None,
                )
            )
        return 42, queue


def client_with(fake: FakeAsyncEngine) -> TestClient:
    api.async_engine = fake
    return TestClient(api.app)


def test_health_and_models():
    client = client_with(FakeAsyncEngine([]))
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}

    r = client.get("/v1/models")
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == api.MODEL_NAME


def test_chat_completion_non_streaming():
    fake = FakeAsyncEngine(["Hello", ", ", "world"], finish_reason="stop")
    client = client_with(fake)
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 16,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == api.MODEL_NAME
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello, world"}
    assert choice["finish_reason"] == "stop"
    # usage accounting: 2 prompt messages -> 2 fake prompt tokens, 3 completion
    assert body["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    # the endpoint must have rendered the chat template and submitted token ids
    assert fake.last_chat is False
    assert fake.rendered_messages[0]["role"] == "system"


def test_chat_completion_length_finish_reason():
    fake = FakeAsyncEngine(["a", "b"], finish_reason="length")
    client = client_with(fake)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "go"}]},
    )
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_chat_completion_streaming_sse():
    fake = FakeAsyncEngine(["Hi", "!"], finish_reason="stop")
    client = client_with(fake)
    payloads: list[str] = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                payloads.append(line[len("data: ") :])

    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    # first chunk announces the assistant role
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # content chunks reconstruct the full message
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "Hi!"
    # every chunk is a chat.completion.chunk with a stable id
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # final chunk carries the finish_reason, empty delta
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}


def test_chat_completion_validation_error():
    client = client_with(FakeAsyncEngine([]))
    # empty messages list violates min_length=1
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 422
    # bad role
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "wizard", "content": "x"}]},
    )
    assert r.status_code == 422


def test_chat_completion_engine_rejection():
    fake = FakeAsyncEngine([])
    fake.raise_on_submit = "request needs up to 999 KV blocks"
    client = client_with(fake)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 422
    assert "error" in r.json()
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_legacy_completions_still_works():
    fake = FakeAsyncEngine(["one", "two"], finish_reason="length")
    client = client_with(fake)
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 8})
    body = r.json()
    assert body["text"] == "onetwo"
    assert body["num_tokens"] == 2
    assert body["finish_reason"] == "length"
    assert fake.last_chat is True  # legacy path applies the chat template
