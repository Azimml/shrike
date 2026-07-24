"""Request lifecycle state for the engine."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field


class Status(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0
    ignore_eos: bool = False  # benchmarks: force fixed-length generations


_req_counter = itertools.count()


@dataclass
class Request:
    token_ids: list[int]  # prompt tokens; generated tokens appended in place
    sampling: SamplingParams
    req_id: int = field(default_factory=lambda: next(_req_counter))
    status: Status = Status.WAITING
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0  # tokens whose KV lives in the cache
    block_table: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    spec_len: int = 0  # tentative (unverified) draft tokens at the tail of token_ids

    def __post_init__(self):
        self.num_prompt_tokens = len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens
