"""Regression tests for the codex correctness-review findings (CPU-only)."""

import pytest

from shrike.engine.block_manager import BlockManager
from shrike.engine.request import Request, SamplingParams, Status
from shrike.engine.scheduler import Scheduler


def make_req(tokens, max_new=8, temperature=0.0):
    return Request(
        token_ids=list(tokens),
        sampling=SamplingParams(max_new_tokens=max_new, temperature=temperature),
    )


def test_failed_admission_releases_prefix_cache_refs():
    """match_prefix revives cached blocks (ref++); if the tail can't be
    allocated the refs must be rolled back, not left pinned forever."""
    bm = BlockManager(num_blocks=2, block_size=4)
    a = make_req(list(range(5)))  # 2 blocks
    bm.append_blocks(a, 5)
    a.num_computed_tokens = 5
    bm.register_full_blocks(a)
    bm.free(a)

    sched = Scheduler(bm, max_tokens_per_step=64)
    # hog one block so the cache-hit request can't fit its tail
    hog = make_req([50, 51, 52, 53])
    bm.append_blocks(hog, 4)
    assert bm.num_free == 1

    b = make_req(list(range(5)) + [99, 98, 97])  # hits the cached first block
    sched.add(b)
    batch = sched.schedule()
    assert batch == []  # can't admit
    # the revived cache ref must have been rolled back
    assert b.block_table == [] and b.num_computed_tokens == 0
    assert bm.num_free == 1  # cached block back on the free queue


def test_spec_drafts_dropped_instead_of_self_preempt():
    """Optional drafts must not force preemption when plain decode fits."""
    bm = BlockManager(num_blocks=1, block_size=4, enable_prefix_caching=False)
    sched = Scheduler(bm, max_tokens_per_step=64, spec_ngram=1, spec_k=4)
    req = make_req([7, 7, 7], max_new=2)
    sched.add(req)
    batch = sched.schedule()  # prefill: 3 tokens, 1 block
    sched.postprocess(batch)
    req.token_ids.append(7)  # sampled token fills the block; drafts would
    # need a 2nd block that doesn't exist

    batch = sched.schedule()
    # decode fits in the existing block; drafts would need a 2nd block and
    # must be dropped, NOT trigger self-preemption
    assert [(r.req_id, n) for r, n in batch] == [(req.req_id, 1)]
    assert req.spec_len == 0
    assert sched.num_preemptions == 0
    assert req.status is Status.RUNNING


def test_no_spec_for_resumed_request_with_history():
    """A preempted request re-decoding generated history must not get
    drafts appended (they would advance unverified)."""
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    sched = Scheduler(bm, max_tokens_per_step=64, spec_ngram=1, spec_k=2)
    req = make_req([1, 2], max_new=8)
    sched.add(req)
    sched.postprocess(sched.schedule())  # prefill
    req.token_ids.extend([1, 2])  # two generated tokens (repetitive history)
    req.num_computed_tokens = 3  # caught up through token 3 of 4

    # simulate preemption/resume: computed reset, prompt recomputed
    req.num_computed_tokens = 2  # NOT caught up: token_ids has 4, computed 2
    batch = sched.schedule()
    got = {r.req_id: n for r, n in batch}
    assert got == {req.req_id: 1}  # plain re-decode of history, no drafts
    assert req.spec_len == 0 and req.num_tokens == 4


@pytest.mark.integration
def test_oversized_and_empty_requests_rejected(model_dir):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from shrike.engine.engine import LLMEngine

    engine = LLMEngine(model_dir, gpu_mem_util=0.3)
    with pytest.raises(ValueError, match="empty"):
        engine.add_request([], SamplingParams())
    with pytest.raises(ValueError, match="never be scheduled"):
        engine.add_request([1, 2, 3], SamplingParams(max_new_tokens=10_000_000))
