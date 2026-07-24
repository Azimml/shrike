"""Edge-case unit tests for the paged allocator and prefix cache (CPU only).

Complements test_block_manager.py's end-to-end flow with focused checks on the
chain hash, the capacity boundary, ref counting, and the caching-disabled path.
"""

from shrike.engine.block_manager import BlockManager
from shrike.engine.request import Request, SamplingParams


def _req(tokens) -> Request:
    return Request(token_ids=list(tokens), sampling=SamplingParams(max_new_tokens=8))


def test_chain_hash_is_deterministic_and_order_sensitive():
    assert BlockManager.chain_hash(None, [1, 2, 3]) == BlockManager.chain_hash(None, [1, 2, 3])
    # different previous-hash context must yield a different block hash, so two
    # blocks with identical tokens but different prefixes never collide
    h = BlockManager.chain_hash(None, [1, 2, 3])
    assert BlockManager.chain_hash(h, [4, 5, 6]) != BlockManager.chain_hash(None, [4, 5, 6])


def test_can_append_boundary():
    bm = BlockManager(num_blocks=2, block_size=4)  # 8 slots total
    req = _req([])
    assert bm.blocks_needed(req, 8) == 2
    assert bm.can_append(req, 8) is True  # exactly fills the pool
    assert bm.can_append(req, 9) is False  # one token over


def test_prefix_caching_disabled_is_inert():
    bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_caching=False)
    req = _req(range(9))
    bm.append_blocks(req, 9)
    req.num_computed_tokens = 9
    bm.register_full_blocks(req)  # no-op when disabled
    assert bm.match_prefix(list(range(9))) == ([], 0)


def test_release_decrements_refcount_and_reveals_free_block():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = _req(range(9))
    bm.append_blocks(a, 9)
    a.num_computed_tokens = 9
    bm.register_full_blocks(a)
    bm.free(a)  # ref_count -> 0, blocks return to free list (hashes kept)

    b = _req(range(9))
    matched, _ = bm.match_prefix(b.token_ids)  # revives a shared block: ref 0 -> 1
    assert bm.blocks[matched[0]].ref_count == 1
    bm.release([matched[0]])
    assert bm.blocks[matched[0]].ref_count == 0  # back on the free list


def test_never_matches_the_whole_prompt():
    # match_prefix must leave >=1 token to compute so the forward pass can
    # produce logits; a prompt that is an exact multiple of block_size still
    # keeps its final block uncached.
    bm = BlockManager(num_blocks=4, block_size=4)
    a = _req(range(8))  # exactly two full blocks
    bm.append_blocks(a, 8)
    a.num_computed_tokens = 8
    bm.register_full_blocks(a)
    bm.free(a)

    _, cached = bm.match_prefix(list(range(8)))
    assert cached == 4  # only the first block matched, not both
