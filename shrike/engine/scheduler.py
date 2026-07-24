"""Iteration-level scheduler: continuous batching (Orca, OSDI '22) with a
Sarathi-Serve (OSDI '24) token budget and chunked prefill.

Every step:
  1. RUNNING sequences are served first. Decodes cost 1 budget token;
     partially-prefilled sequences continue with a chunk that fits the
     remaining budget.
  2. Remaining budget admits WAITING sequences (FCFS), chunking their prompts.
  3. If a decode can't get a KV block, the newest running sequence is
     preempted (blocks freed, recomputed later) — discard-and-recompute
     preemption, as in vLLM.

Chunked prefill keeps every step's token count bounded, so a long prompt
can't stall in-flight decodes — this is what tames p99 inter-token latency.
"""

from __future__ import annotations

from collections import deque

from shrike.engine.block_manager import BlockManager
from shrike.engine.request import Request, Status
from shrike.engine.spec_ngram import propose


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_tokens_per_step: int = 512,
        max_running: int = 256,
        spec_ngram: int = 0,  # n-gram size for prompt-lookup speculation, 0 = off
        spec_k: int = 4,  # max draft tokens per step
    ) -> None:
        self.bm = block_manager
        self.max_tokens_per_step = max_tokens_per_step
        self.max_running = max_running
        self.spec_ngram = spec_ngram
        self.spec_k = spec_k
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.num_preemptions = 0

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt_for(self, req: Request) -> bool:
        """Free blocks by preempting the newest running request. Returns False
        if req itself became the victim (caller must skip it this step)."""
        victim = self.running.pop()
        self.num_preemptions += 1
        self.bm.free(victim)
        self._drop_drafts(victim)
        victim.num_computed_tokens = 0
        victim.status = Status.WAITING
        self.waiting.appendleft(victim)
        return victim is not req

    @staticmethod
    def _drop_drafts(req: Request) -> None:
        if req.spec_len:
            del req.token_ids[-req.spec_len :]
            req.spec_len = 0

    def schedule(self) -> list[tuple[Request, int]]:
        batch: list[tuple[Request, int]] = []
        prefill_batch: list[tuple[Request, int]] = []
        budget = self.max_tokens_per_step

        # 1. running sequences: decodes and in-progress prefills
        for req in list(self.running):
            if budget == 0:
                break
            if req.status is not Status.RUNNING:  # preempted earlier this step
                continue
            if req.prefill_done:
                n_new = 1
                # propose drafts only when fully caught up (a resumed
                # preempted request first re-decodes its generated history
                # one token per step; speculating there would advance
                # unverified tokens — no verification span gets produced)
                caught_up = req.num_computed_tokens == req.num_tokens - 1
                if self.spec_ngram and caught_up and req.sampling.temperature == 0.0 and budget > 1:
                    drafts = propose(req.token_ids, self.spec_ngram, self.spec_k)
                    req.spec_len = min(len(drafts), budget - 1)
                    if req.spec_len:
                        req.token_ids.extend(drafts[: req.spec_len])
                        n_new = 1 + req.spec_len
            else:
                n_new = min(req.num_prompt_tokens - req.num_computed_tokens, budget)
            while not self.bm.can_append(req, n_new):
                if req.spec_len:  # drafts are optional: drop them before
                    self._drop_drafts(req)  # preempting anyone (incl. self)
                    n_new = 1
                    continue
                if not self._preempt_for(req):
                    n_new = 0
                    break
            if n_new == 0:
                continue
            self.bm.append_blocks(req, n_new)
            (batch if n_new == 1 else prefill_batch).append((req, n_new))
            budget -= n_new

        # 2. admit waiting sequences with prefill chunks
        while self.waiting and budget > 0 and len(self.running) < self.max_running:
            req = self.waiting[0]
            if not req.block_table and self.bm.enable_prefix_caching:
                cached_blocks, cached_tokens = self.bm.match_prefix(
                    req.token_ids[: req.num_prompt_tokens]
                )
                req.block_table.extend(cached_blocks)
                req.num_computed_tokens = cached_tokens
            n_new = min(req.num_prompt_tokens - req.num_computed_tokens, budget)
            if not self.bm.can_append(req, n_new):
                if req.block_table:  # roll back revived prefix-cache refs so
                    self.bm.release(req.block_table)  # they don't stay pinned
                    req.block_table = []
                    req.num_computed_tokens = 0
                break  # FCFS: don't skip ahead of the head request
            self.waiting.popleft()
            self.bm.append_blocks(req, n_new)
            req.status = Status.RUNNING
            self.running.append(req)
            (batch if n_new == 1 else prefill_batch).append((req, n_new))
            budget -= n_new

        return batch + prefill_batch  # decodes first (ModelRunner convention)

    def postprocess(
        self, batch: list[tuple[Request, int]], advances: dict[int, int] | None = None
    ) -> None:
        """Advance computed-token counts (speculative requests advance by
        their verified length, not the scheduled draft length); register
        full blocks for prefix cache."""
        for req, n_new in batch:
            req.num_computed_tokens += (advances or {}).get(req.req_id, n_new)
            self.bm.register_full_blocks(req)

    def finish(self, req: Request, reason: str) -> None:
        req.status = Status.FINISHED
        req.finish_reason = reason
        self.running.remove(req)
        self.bm.free(req)
