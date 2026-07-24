"""CPU-only unit tests for the paged allocator + prefix cache + scheduler."""

from shrike.engine.block_manager import BlockManager
from shrike.engine.request import Request, SamplingParams, Status


def make_req(tokens: list[int]) -> Request:
    return Request(token_ids=list(tokens), sampling=SamplingParams(max_new_tokens=8))


def test_allocate_and_free():
    bm = BlockManager(num_blocks=8, block_size=4)
    req = make_req(list(range(10)))  # 10 tokens -> 3 blocks
    assert bm.blocks_needed(req, 10) == 3
    bm.append_blocks(req, 10)
    assert len(req.block_table) == 3 and bm.num_free == 5
    req.num_computed_tokens = 10
    # 11th token crosses into a new block only at 13 (3 blocks hold 12)
    assert bm.blocks_needed(req, 1) == 0
    req.num_computed_tokens = 12
    assert bm.blocks_needed(req, 1) == 1
    bm.free(req)
    assert bm.num_free == 8 and req.block_table == []


def test_prefix_cache_hit_and_eviction():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = make_req(list(range(9)))  # blocks: [0..3][4..7][8]
    bm.append_blocks(a, 9)
    a.num_computed_tokens = 9
    bm.register_full_blocks(a)  # hashes 2 full blocks
    bm.free(a)

    b = make_req(list(range(9)))  # same prompt
    matched, cached = bm.match_prefix(b.token_ids)
    assert cached == 8 and len(matched) == 2  # never matches the whole prompt
    b.block_table.extend(matched)
    b.num_computed_tokens = cached
    bm.append_blocks(b, 1)  # just the tail token
    assert len(b.block_table) == 3

    # c shares only the first block's worth of tokens
    c = make_req(list(range(4)) + [99, 98, 97, 96, 95])
    matched_c, cached_c = bm.match_prefix(c.token_ids)
    c.block_table.extend(matched_c)
    c.num_computed_tokens = cached_c
    assert cached_c == 4 and len(matched_c) == 1
    assert bm.blocks[matched_c[0]].ref_count == 2  # shared with b

    bm.free(b)
    bm.free(c)
    # exhaust the pool with an unrelated request -> cached blocks get evicted
    d = make_req(list(range(100, 116)))
    bm.append_blocks(d, 16)
    assert bm.num_free == 0
    e_matched, e_cached = bm.match_prefix(list(range(9)))
    assert e_cached == 0  # everything evicted


def test_scheduler_chunked_prefill_and_preemption():
    from shrike.engine.scheduler import Scheduler

    bm = BlockManager(num_blocks=6, block_size=4, enable_prefix_caching=False)
    sched = Scheduler(bm, max_tokens_per_step=8, max_running=4)
    long = make_req(list(range(20)))  # needs 5 blocks for prompt
    short = make_req(list(range(200, 206)))  # 6 tokens
    sched.add(long)
    sched.add(short)

    batch = sched.schedule()  # budget 8: long gets an 8-token chunk, nothing left
    assert [(r.req_id, n) for r, n in batch] == [(long.req_id, 8)]
    sched.postprocess(batch)
    assert long.num_computed_tokens == 8

    batch = sched.schedule()  # long continues 8, budget exhausted
    sched.postprocess(batch)
    batch = sched.schedule()  # long finishes prompt (4), short admitted (4)
    sched.postprocess(batch)
    assert long.prefill_done
    assert short.num_computed_tokens == 4

    # long's first decode token needs a 6th block; the pool is exhausted, so
    # the newest running request (short) must be preempted for it
    long.token_ids.append(1000)  # simulate sampled token
    batch = sched.schedule()
    got = {r.req_id: n for r, n in batch}
    assert got == {long.req_id: 1}
    assert sched.num_preemptions == 1
    assert short.status is Status.WAITING and short.num_computed_tokens == 0
    sched.postprocess(batch)

    # long finishes -> its blocks free up -> short gets readmitted and
    # recomputes its prompt from scratch (discard-and-recompute)
    sched.finish(long, "stop")
    batch = sched.schedule()
    got = {r.req_id: n for r, n in batch}
    assert got == {short.req_id: 6}  # full prompt recompute in one chunk
    sched.postprocess(batch)
    assert short.prefill_done
