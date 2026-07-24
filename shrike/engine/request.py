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

    def __post_init__(self) -> None:
        # Validate at construction so bad params fail fast at the call site
        # rather than deep in the sampler (which would divide by temperature
        # or mask an empty nucleus). Mirrors the server's pydantic bounds so
        # the offline and HTTP paths reject the same inputs.
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


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
