"""Triton paged-attention decode kernel.

One program per (sequence, query head): the kernel walks the sequence's
block table tile-by-tile, gathering K/V slots directly from the paged pool
and folding them into an online softmax — no contiguous KV copy, no
expanded-GQA materialization. This is the fused alternative to the
grouped-einsum decode path in model_runner.py (select with
LLMEngine(attention_backend="triton")).

Decode only (q_len == 1). Prefill keeps the sliced-SDPA path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr,  # [B, H, D]
    k_ptr,  # [num_slots, H_kv, D]
    v_ptr,  # [num_slots, H_kv, D]
    tables_ptr,  # [B, max_blocks] int
    ctx_ptr,  # [B] int
    out_ptr,  # [B, H, D]
    stride_qb,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_tb,
    stride_ob,
    stride_oh,
    scale,
    GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_head = h // GROUPS

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_d).to(tl.float32)

    ctx_len = tl.load(ctx_ptr + b)
    m_i = float("-inf")  # running max
    l_i = 0.0  # running sum of exp
    acc = tl.zeros([D], dtype=tl.float32)

    for start_n in range(0, ctx_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < ctx_len
        block_ids = tl.load(tables_ptr + b * stride_tb + offs_n // BLOCK_SIZE, mask=n_mask, other=0)
        slots = block_ids * BLOCK_SIZE + offs_n % BLOCK_SIZE

        k = tl.load(
            k_ptr + slots[:, None] * stride_ks + kv_head * stride_kh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(n_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        v = tl.load(
            v_ptr + slots[:, None] * stride_ks + kv_head * stride_kh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, out.to(out_ptr.dtype.element_ty))


def triton_paged_decode(
    q: torch.Tensor,  # [B, H, D]
    k_pool: torch.Tensor,  # [num_slots, H_kv, D]
    v_pool: torch.Tensor,  # [num_slots, H_kv, D]
    block_tables: torch.Tensor,  # [B, max_blocks] long, 0-padded
    ctx_lens: torch.Tensor,  # [B] long
    block_size: int,
) -> torch.Tensor:
    b, h, d = q.shape
    h_kv = k_pool.shape[1]
    out = torch.empty_like(q)
    q = q.contiguous()
    block_tables = block_tables.contiguous()
    _paged_decode_kernel[(b, h)](
        q,
        k_pool,
        v_pool,
        block_tables,
        ctx_lens,
        out,
        q.stride(0),
        q.stride(1),
        k_pool.stride(0),
        k_pool.stride(1),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        d**-0.5,
        GROUPS=h // h_kv,
        BLOCK_SIZE=block_size,
        BLOCK_N=128,
        D=d,
    )
    return out
