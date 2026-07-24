"""Paged KV cache allocator (PagedAttention, SOSP '23) with hash-based
prefix caching (vLLM v1 automatic prefix caching).

The KV pool is a fixed set of `block_size`-token blocks. Sequences hold block
tables (lists of block ids) instead of contiguous cache tensors, eliminating
fragmentation and enabling copy-free prefix sharing via ref counts.

Prefix caching: every *full* block is keyed by a chain hash
    h_i = hash(h_{i-1}, tokens_in_block_i)
Freed blocks keep their hash and go to the free list LRU-ordered; a cache hit
on allocation revives the block (ref_count 0 -> 1) instead of recomputing it.
"""

from __future__ import annotations

from collections import deque

from shrike.engine.request import Request


class Block:
    __slots__ = ("block_id", "ref_count", "hash")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_queue: deque[int] = deque(range(num_blocks))  # LRU: popleft oldest
        self.hash_to_block: dict[int, int] = {}
        self.cache_hit_blocks = 0
        self.cache_query_blocks = 0

    @property
    def num_free(self) -> int:
        return len(self.free_queue)

    @staticmethod
    def chain_hash(prev_hash: int | None, tokens: list[int]) -> int:
        return hash((prev_hash, tuple(tokens)))

    def _pop_free_block(self) -> Block:
        block = self.blocks[self.free_queue.popleft()]
        if block.hash is not None:  # evict stale cache entry
            if self.hash_to_block.get(block.hash) == block.block_id:
                del self.hash_to_block[block.hash]
            block.hash = None
        block.ref_count = 1
        return block

    def blocks_needed(self, req: Request, num_new_tokens: int) -> int:
        total = req.num_computed_tokens + num_new_tokens
        return max(0, -(-total // self.block_size) - len(req.block_table))

    def can_append(self, req: Request, num_new_tokens: int) -> bool:
        return self.blocks_needed(req, num_new_tokens) <= self.num_free

    def append_blocks(self, req: Request, num_new_tokens: int) -> None:
        for _ in range(self.blocks_needed(req, num_new_tokens)):
            req.block_table.append(self._pop_free_block().block_id)

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest cached full-block prefix -> (block_ids, num_cached_tokens).

        Referenced blocks get ref_count++ (revived from the free list if
        needed); caller must treat those tokens as already computed.
        """
        if not self.enable_prefix_caching:
            return [], 0
        matched: list[int] = []
        prev_hash: int | None = None
        # never match the *entire* prompt: leave >=1 token to compute so the
        # forward pass produces logits for sampling
        num_full = (len(token_ids) - 1) // self.block_size
        for i in range(num_full):
            chunk = token_ids[i * self.block_size : (i + 1) * self.block_size]
            prev_hash = self.chain_hash(prev_hash, chunk)
            self.cache_query_blocks += 1
            block_id = self.hash_to_block.get(prev_hash)
            if block_id is None:
                break
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self.free_queue.remove(block_id)  # revive
            block.ref_count += 1
            matched.append(block_id)
            self.cache_hit_blocks += 1
        return matched, len(matched) * self.block_size

    def register_full_blocks(self, req: Request) -> None:
        """Hash req's newly-completed full blocks so future requests can reuse them."""
        if not self.enable_prefix_caching:
            return
        prev_hash: int | None = None
        num_full = req.num_computed_tokens // self.block_size
        for i in range(num_full):
            chunk = req.token_ids[i * self.block_size : (i + 1) * self.block_size]
            prev_hash = self.chain_hash(prev_hash, chunk)
            block = self.blocks[req.block_table[i]]
            if block.hash is None:
                block.hash = prev_hash
                self.hash_to_block.setdefault(prev_hash, block.block_id)

    def release(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_queue.append(block_id)  # keep hash: still cache-hittable

    def free(self, req: Request) -> None:
        self.release(req.block_table)
        req.block_table = []
