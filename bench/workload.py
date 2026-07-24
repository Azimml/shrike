"""Canonical benchmark workload, shared by every rung.

Realistic serving traffic has *variable* output lengths — that is the entire
motivation for continuous batching (Orca, OSDI '22): a static batch runs
until its longest member finishes, wasting decode compute on rows that are
already done. Every rung draws the same seeded per-request lengths so the
comparison is apples-to-apples; static batching additionally reports
"useful" throughput (only each row's own target tokens count).
"""

from __future__ import annotations

import random

MIN_NEW_TOKENS = 32
MAX_NEW_TOKENS = 256
_SEED = 20260718


def target_lengths(num_requests: int) -> list[int]:
    rng = random.Random(_SEED)
    return [rng.randint(MIN_NEW_TOKENS, MAX_NEW_TOKENS) for _ in range(num_requests)]
