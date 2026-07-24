"""Asyncio bridge over the synchronous LLMEngine.

One background task drives engine.step() in a thread-pool executor (the GPU
forward releases the GIL poorly, but the executor keeps the event loop
responsive for request intake and SSE streaming). Each request gets an
asyncio.Queue; step outputs are fanned out to the owning queue.

This is the 'async serving layer': request queue in, token streams out,
hundreds of concurrent HTTP requests multiplexed onto one GPU decode loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast

from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams


@dataclass
class TokenEvent:
    token_id: int
    text: str
    finished: bool
    finish_reason: str | None


class AsyncEngine:
    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self.queues: dict[int, asyncio.Queue[TokenEvent]] = {}
        self.decoded: dict[int, list[int]] = {}  # req_id -> undecoded token buffer
        self.first_token_at: dict[int, float | None] = {}
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def submit(
        self,
        prompt: str | list[int],
        sampling: SamplingParams,
        chat: bool = True,
    ) -> tuple[int, asyncio.Queue[TokenEvent]]:
        req_id = self.engine.add_request(prompt, sampling, chat=chat)
        queue: asyncio.Queue[TokenEvent] = asyncio.Queue()
        self.queues[req_id] = queue
        self.decoded[req_id] = []
        self._wakeup.set()
        return req_id, queue

    def render_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template to OpenAI-style messages, returning
        prompt token ids ready for submit(..., chat=False)."""
        # tokenize=True (the default) => a mapping whose "input_ids" is a
        # token-id list; the transformers return type widens to str for the
        # tokenize=False case, hence the cast.
        encoded = cast(
            "dict[str, list[int]]",
            self.engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True),
        )
        return list(encoded["input_ids"])

    def _decode_incremental(self, req_id: int, token_id: int) -> str:
        """Buffer tokens until they decode to text without a dangling
        replacement char (multi-byte glyphs span BPE tokens)."""
        buf = self.decoded[req_id]
        buf.append(token_id)
        text = cast("str", self.engine.tokenizer.decode(buf, skip_special_tokens=True))
        if text.endswith("�"):
            return ""
        buf.clear()
        return text

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.engine.has_work:
                self._wakeup.clear()
                await self._wakeup.wait()
            try:
                outputs = await loop.run_in_executor(None, self.engine.step)
            except Exception:
                # a silent task death wedges every open SSE stream; fail loud
                import traceback

                traceback.print_exc()
                for open_queue in self.queues.values():
                    await open_queue.put(TokenEvent(0, "", True, "engine_error"))
                self.queues.clear()
                self.decoded.clear()
                raise
            for out in outputs:
                queue = self.queues.get(out.req_id)
                if queue is None:
                    continue
                text = self._decode_incremental(out.req_id, out.token_id)
                await queue.put(TokenEvent(out.token_id, text, out.finished, out.finish_reason))
                if out.finished:
                    del self.queues[out.req_id]
                    del self.decoded[out.req_id]
            await asyncio.sleep(0)  # yield to intake/streaming coroutines
