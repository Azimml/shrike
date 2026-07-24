"""Unit tests for scheduler admission policy (pure Python, no model/GPU).

Complements the chunked-prefill/preemption test in test_block_manager.py by
pinning down the admission-side invariants: has_work, the max_running cap,
FCFS head-of-line blocking, and draft dropping.
"""

from shrike.engine.block_manager import BlockManager
from shrike.engine.request import Request, SamplingParams, Status
from shrike.engine.scheduler import Scheduler


def _req(tokens) -> Request:
    return Request(token_ids=list(tokens), sampling=SamplingParams(max_new_tokens=8))


def _bm(num_blocks=8, block_size=4) -> BlockManager:
    return BlockManager(num_blocks, block_size, enable_prefix_caching=False)


def test_has_work_and_empty_schedule():
    sched = Scheduler(_bm())
    assert sched.has_work is False
    assert sched.schedule() == []
    sched.add(_req(range(4)))
    assert sched.has_work is True


def test_max_running_caps_concurrency():
    # plenty of KV blocks, but max_running=1 must admit only one sequence
    sched = Scheduler(_bm(num_blocks=20), max_tokens_per_step=64, max_running=1)
    a, b = _req(range(4)), _req(range(100, 104))
    sched.add(a)
    sched.add(b)
    batch = sched.schedule()
    assert [r.req_id for r, _ in batch] == [a.req_id]
    assert len(sched.running) == 1 and len(sched.waiting) == 1
    assert b.status is Status.WAITING


def test_fcfs_head_of_line_blocking():
    # pool has 2 blocks (8 slots). The head request needs 3 blocks and can
    # never be admitted right now; FCFS must NOT skip ahead to the smaller one.
    sched = Scheduler(_bm(num_blocks=2), max_tokens_per_step=64, max_running=8)
    big = _req(range(12))  # 3 blocks needed > 2 free
    small = _req(range(4))  # would fit on its own
    sched.add(big)
    sched.add(small)
    assert sched.schedule() == []  # blocked behind big
    assert len(sched.running) == 0
    assert small.status is Status.WAITING


def test_drop_drafts_truncates_speculative_tail():
    req = _req(range(4))
    req.token_ids.extend([9, 9])  # two tentative draft tokens
    req.spec_len = 2
    Scheduler._drop_drafts(req)
    assert req.spec_len == 0
    assert req.token_ids == [0, 1, 2, 3]  # drafts removed in place


def test_drop_drafts_is_noop_without_drafts():
    req = _req(range(4))
    Scheduler._drop_drafts(req)  # spec_len == 0
    assert req.token_ids == [0, 1, 2, 3]
