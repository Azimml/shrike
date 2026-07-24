"""LLMEngine: ties model runner + scheduler + sampler into a step loop.

step() = schedule -> one batched forward -> sample -> append/finish.
Synchronous by design; the async serving layer drives it from a background
task and fans tokens out to per-request queues.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import cast

from transformers import AutoTokenizer

from shrike.engine.block_manager import BlockManager
from shrike.engine.model_runner import ModelRunner
from shrike.engine.request import Request, SamplingParams
from shrike.engine.sampler import sample
from shrike.engine.scheduler import Scheduler
from shrike.models.qwen import QwenForCausalLM


@dataclass
class StepOutput:
    req_id: int
    token_id: int
    finished: bool
    finish_reason: str | None


@dataclass
class EngineStats:
    steps: int = 0
    tokens_generated: int = 0
    spec_drafted: int = 0
    spec_accepted: int = 0
    started_at: float = field(default_factory=time.perf_counter)


class LLMEngine:
    def __init__(
        self,
        model_dir: str,
        block_size: int = 16,
        gpu_mem_util: float = 0.85,
        max_tokens_per_step: int = 512,
        max_running: int = 256,
        enable_prefix_caching: bool = True,
        spec_ngram: int = 0,
        spec_k: int = 4,
        attention_backend: str = "einsum",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = QwenForCausalLM.load(model_dir)
        self.runner = ModelRunner(self.model, block_size, gpu_mem_util, attention_backend)
        self.block_manager = BlockManager(self.runner.num_blocks, block_size, enable_prefix_caching)
        self.scheduler = Scheduler(
            self.block_manager, max_tokens_per_step, max_running, spec_ngram, spec_k
        )
        self.eos_ids = set(self.model.cfg.eos_token_ids)
        self.requests: dict[int, Request] = {}
        self.stats = EngineStats()

    def add_request(
        self, prompt: str | list[int], sampling: SamplingParams, chat: bool = True
    ) -> int:
        token_ids: list[int]
        if isinstance(prompt, str):
            if chat:
                encoded = cast(
                    "dict[str, list[int]]",
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                    ),
                )
                token_ids = list(encoded["input_ids"])
            else:
                token_ids = list(self.tokenizer(prompt).input_ids)
        else:
            token_ids = list(prompt)
        if not token_ids:
            raise ValueError(
                "empty prompt: a request must have at least one token "
                "(the prompt tokenized to zero tokens)"
            )
        if sampling.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {sampling.max_new_tokens}")
        bs = self.block_manager.block_size
        worst_case_blocks = -(-(len(token_ids) + sampling.max_new_tokens) // bs)
        if worst_case_blocks > self.block_manager.num_blocks:
            raise ValueError(
                f"request needs up to {worst_case_blocks} KV blocks "
                f"({len(token_ids)} prompt + {sampling.max_new_tokens} generated tokens, "
                f"block_size={bs}) but the pool has only {self.block_manager.num_blocks}; "
                "it could never be scheduled — shorten the prompt, lower max_new_tokens, "
                "or raise gpu_mem_util"
            )
        req = Request(token_ids=token_ids, sampling=sampling)
        self.requests[req.req_id] = req
        self.scheduler.add(req)
        return req.req_id

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    def step(self) -> list[StepOutput]:
        batch = self.scheduler.schedule()
        if not batch:
            return []
        logits, spans = self.runner.run(batch)
        advances: dict[int, int] = {}
        emit: list[tuple[Request, list[int]]] = []  # newly committed tokens per req

        singles = [(req, off) for req, off, n in spans if n == 1]
        if singles:
            rows = logits[[off for _, off in singles]]
            for (req, _), token in zip(singles, sample(rows, [r for r, _ in singles]), strict=True):
                req.token_ids.append(token)
                emit.append((req, [token]))

        for req, off, n in spans:  # speculative chunks: verify drafts greedily
            if n == 1:
                continue
            k = req.spec_len
            preds = logits[off : off + n].argmax(-1).tolist()  # n == k + 1 rows
            committed = req.num_tokens - k
            drafts = req.token_ids[committed:]
            accepted = 0
            while accepted < k and drafts[accepted] == preds[accepted]:
                accepted += 1
            del req.token_ids[committed + accepted :]
            req.token_ids.append(preds[accepted])  # bonus token: free correction
            req.spec_len = 0
            advances[req.req_id] = 1 + accepted
            self.stats.spec_drafted += k
            self.stats.spec_accepted += accepted
            emit.append((req, req.token_ids[committed:]))

        # postprocess BEFORE finishing: finish() frees block tables, and
        # register_full_blocks must see the request's final token state
        self.scheduler.postprocess(batch, advances)

        outputs: list[StepOutput] = []
        for req, tokens in emit:
            base_generated = req.num_generated - len(tokens)
            for i, token in enumerate(tokens):
                self.stats.tokens_generated += 1
                finished, reason = False, None
                if token in self.eos_ids and not req.sampling.ignore_eos:
                    finished, reason = True, "stop"
                elif base_generated + i + 1 >= req.sampling.max_new_tokens:
                    finished, reason = True, "length"
                outputs.append(StepOutput(req.req_id, token, finished, reason))
                if finished:
                    assert reason is not None  # set together with finished above
                    del req.token_ids[req.num_prompt_tokens + base_generated + i + 1 :]
                    self.scheduler.finish(req, reason)
                    del self.requests[req.req_id]
                    break
        self.stats.steps += 1
        return outputs

    def generate(
        self,
        prompts: list[str | list[int]],
        sampling: SamplingParams | list[SamplingParams],
        chat: bool = True,
    ) -> list[list[int]]:
        """Offline batch API for benchmarks: returns generated token ids per prompt."""
        if isinstance(sampling, SamplingParams):
            sampling = [sampling] * len(prompts)
        id_map = {
            self.add_request(p, s, chat): i
            for i, (p, s) in enumerate(zip(prompts, sampling, strict=True))
        }
        results: list[list[int]] = [[] for _ in prompts]
        while self.has_work:
            for out in self.step():
                results[id_map[out.req_id]].append(out.token_id)
        return results

    def metrics(self) -> dict:
        bm, sched = self.block_manager, self.scheduler
        elapsed = time.perf_counter() - self.stats.started_at
        return {
            "steps": self.stats.steps,
            "tokens_generated": self.stats.tokens_generated,
            "avg_tokens_per_sec": round(self.stats.tokens_generated / elapsed, 1),
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "kv_blocks_total": bm.num_blocks,
            "kv_blocks_free": bm.num_free,
            "preemptions": sched.num_preemptions,
            "prefix_cache_hit_rate": round(bm.cache_hit_blocks / max(1, bm.cache_query_blocks), 3),
            "spec_drafted": self.stats.spec_drafted,
            "spec_accepted": self.stats.spec_accepted,
            "spec_acceptance_rate": round(
                self.stats.spec_accepted / max(1, self.stats.spec_drafted), 3
            ),
        }
