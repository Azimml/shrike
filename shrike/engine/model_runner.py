"""Batched forward over a paged KV cache.

Each engine step runs ONE flat token batch through the model (Orca-style
iteration-level batching): all scheduled sequences' new tokens are
concatenated to shape [N_tokens]. Dense layers (embeddings, projections, MLP)
are sequence-agnostic; only attention consults the batch metadata.

Token order convention: [all decode tokens (1 per seq)] + [prefill chunks].

Attention is pure PyTorch: new K/V are scattered into the paged pool via a
slot mapping, then per-sequence context is gathered from block tables and fed
to scaled_dot_product_attention. A production engine fuses this into a paged
attention kernel (vLLM SOSP '23); the gather is our deliberate simplicity
trade-off and is documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from shrike.engine.request import Request
from shrike.models.qwen import QwenConfig, QwenForCausalLM


@dataclass
class BatchMeta:
    slot_mapping: torch.Tensor  # [N] long: flat pool slot for each new token
    num_decode_seqs: int
    decode_block_tables: torch.Tensor | None  # [B_dec, max_blocks] long, 0-padded
    decode_ctx_lens: torch.Tensor | None  # [B_dec] long, includes the new token
    # per prefill seq: (q_start, q_len, ctx_len, block_table [num_blocks] long)
    prefill: list[tuple[int, int, int, torch.Tensor]] = field(default_factory=list)


class PagedKVBackend:
    """KV pool: [num_layers, 2, num_slots, num_kv_heads, head_dim]."""

    def __init__(
        self,
        cfg: QwenConfig,
        num_blocks: int,
        block_size: int,
        device: str = "cuda",
        attention_backend: str = "einsum",  # "einsum" | "triton"
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.attention_backend = attention_backend
        self.pool = torch.zeros(
            cfg.num_layers,
            2,
            num_blocks * block_size,
            cfg.num_kv_heads,
            cfg.head_dim,
            dtype=cfg.dtype,
            device=device,
        )
        self.meta: BatchMeta | None = None  # set by ModelRunner before each forward

    def run(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        meta = self.meta
        assert meta is not None, "ModelRunner must set backend.meta before each forward"
        k_pool, v_pool = self.pool[layer_idx, 0], self.pool[layer_idx, 1]
        k_pool.index_copy_(0, meta.slot_mapping, k)
        v_pool.index_copy_(0, meta.slot_mapping, v)

        out = torch.empty_like(q)
        bs = self.block_size
        k_blocks = k_pool.view(self.num_blocks, bs, *k_pool.shape[1:])
        v_blocks = v_pool.view(self.num_blocks, bs, *v_pool.shape[1:])

        n_dec = meta.num_decode_seqs
        if n_dec:
            # n_dec > 0 guarantees the runner populated the decode metadata
            assert meta.decode_block_tables is not None
            assert meta.decode_ctx_lens is not None
            decode_tables = meta.decode_block_tables
            decode_ctx = meta.decode_ctx_lens
        if n_dec and self.attention_backend == "triton":
            from shrike.engine.triton_attention import triton_paged_decode

            out[:n_dec] = triton_paged_decode(
                q[:n_dec],
                k_pool,
                v_pool,
                decode_tables,
                decode_ctx,
                self.block_size,
            )
        elif n_dec:
            # Chunk the gather and sort by context length: gathering all
            # sequences padded to the global max context allocates ~GBs of
            # transients at high concurrency and stalls the CUDA allocator
            # on a 4GB card. Sorted chunks keep padding tight and transients
            # bounded regardless of batch size.
            order = torch.argsort(decode_ctx)
            ctx_sorted = decode_ctx[order].tolist()
            # Chunk boundaries budgeted by padded token-slots, not sequence
            # count: the masked-SDPA math fallback materializes tensors
            # proportional to chunk_size * padded_ctx, which OOMs a 4GB card
            # at long contexts if the chunk is a fixed 64 sequences.
            chunks = []
            i = 0
            while i < len(ctx_sorted):
                n = 1
                while i + n < len(ctx_sorted) and n < 64 and (n + 1) * ctx_sorted[i + n] <= 32768:
                    n += 1
                chunks.append(order[i : i + n])
                i += n
            h_kv = self.pool.shape[3]
            head_dim = self.pool.shape[4]
            groups = q.shape[1] // h_kv
            for chunk in chunks:
                ctx = decode_ctx[chunk]
                max_blocks = -(-int(ctx.max()) // bs)
                tables = decode_tables[chunk, :max_blocks]
                kg = k_blocks[tables].flatten(1, 2)  # [b, S, H_kv, D]
                vg = v_blocks[tables].flatten(1, 2)
                # Manual grouped-GQA attention: SDPA with a bool mask falls
                # back to the math kernel, which materializes fp32 scores AND
                # KV expanded to all query heads — hundreds of MB at long
                # contexts. The grouped einsum never expands KV; the score
                # tensor is [b, H, S] (q_len is 1 for decode).
                qg = q[chunk].view(len(chunk), h_kv, groups, head_dim)
                scores = torch.einsum("bkgd,bskd->bkgs", qg.float(), kg.float())
                scores *= head_dim**-0.5
                pad = torch.arange(kg.shape[1], device=q.device)[None, :] >= ctx[:, None]
                scores.masked_fill_(pad[:, None, None, :], float("-inf"))
                probs = scores.softmax(-1)
                attn = torch.einsum("bkgs,bskd->bkgd", probs, vg.float())
                out[chunk] = attn.reshape(len(chunk), -1, head_dim).to(out.dtype)

        for q_start, q_len, ctx_len, block_table in meta.prefill:
            kg = k_blocks[block_table].flatten(0, 1)[:ctx_len].transpose(0, 1)[None]  # [1,H_kv,C,D]
            vg = v_blocks[block_table].flatten(0, 1)[:ctx_len].transpose(0, 1)[None]
            # slice queries so the math-fallback score tensor stays bounded
            # (a full 512-token chunk against a long context is ~100MB fp32)
            for qs in range(0, q_len, 128):
                qe = min(qs + 128, q_len)
                qp = q[q_start + qs : q_start + qe].permute(1, 0, 2)[None]  # [1, H, l, D]
                mask = torch.ones(qe - qs, ctx_len, dtype=torch.bool, device=q.device).tril(
                    diagonal=ctx_len - q_len + qs
                )
                attn = F.scaled_dot_product_attention(qp, kg, vg, attn_mask=mask, enable_gqa=True)
                out[q_start + qs : q_start + qe] = attn[0].permute(1, 0, 2)
        return out


class ModelRunner:
    def __init__(
        self,
        model: QwenForCausalLM,
        block_size: int,
        gpu_mem_util: float = 0.85,
        attention_backend: str = "einsum",
    ):
        self.model = model
        self.cfg = model.cfg
        self.block_size = block_size
        free, _total = torch.cuda.mem_get_info()
        block_bytes = (
            self.cfg.num_layers
            * 2
            * block_size
            * self.cfg.num_kv_heads
            * self.cfg.head_dim
            * self.pool_dtype_size()
        )
        self.num_blocks = int(free * gpu_mem_util) // block_bytes
        self.backend = PagedKVBackend(
            self.cfg, self.num_blocks, block_size, attention_backend=attention_backend
        )

    def pool_dtype_size(self) -> int:
        return torch.tensor([], dtype=self.cfg.dtype).element_size()

    @torch.inference_mode()
    def run(
        self, batch: list[tuple[Request, int]]
    ) -> tuple[torch.Tensor, list[tuple[Request, int, int]]]:
        """Execute one step. batch = [(req, num_new_tokens)], decodes first.

        Returns (logits, spans). Logits are computed only for rows that get
        sampled/verified; spans = [(req, offset, n_rows)] indexes into them.
        Normal requests contribute 1 row (their last computed token);
        speculative requests contribute their whole chunk so every draft
        position can be verified.
        """
        device = "cuda"
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        sample_rows: list[int] = []
        spans: list[tuple[Request, int, int]] = []

        decode_tables, decode_ctx = [], []
        prefill_meta: list[tuple[int, int, int, torch.Tensor]] = []
        num_decode_seqs = sum(1 for _, n in batch if n == 1)

        for req, n_new in batch:
            start = req.num_computed_tokens
            toks = req.token_ids[start : start + n_new]
            q_start = len(input_ids)
            input_ids.extend(toks)
            positions.extend(range(start, start + n_new))
            slots.extend(
                req.block_table[p // self.block_size] * self.block_size + p % self.block_size
                for p in range(start, start + n_new)
            )
            ctx_len = start + n_new
            table = torch.tensor(req.block_table, dtype=torch.long, device=device)
            if n_new == 1:
                decode_tables.append(table)
                decode_ctx.append(ctx_len)
            else:
                prefill_meta.append((q_start, n_new, ctx_len, table))
            if ctx_len >= req.num_tokens:  # computed through last token -> sample
                if req.spec_len > 0:  # verify every draft position
                    spans.append((req, len(sample_rows), n_new))
                    sample_rows.extend(range(q_start, q_start + n_new))
                else:
                    spans.append((req, len(sample_rows), 1))
                    sample_rows.append(q_start + n_new - 1)

        meta = BatchMeta(
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
            num_decode_seqs=num_decode_seqs,
            decode_block_tables=(
                torch.nn.utils.rnn.pad_sequence(decode_tables, batch_first=True)
                if decode_tables
                else None
            ),
            decode_ctx_lens=(
                torch.tensor(decode_ctx, dtype=torch.long, device=device) if decode_ctx else None
            ),
            prefill=prefill_meta,
        )
        self.backend.meta = meta
        logits = self.model(
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(positions, dtype=torch.long, device=device),
            self.backend,
            out_rows=torch.tensor(sample_rows, dtype=torch.long, device=device),
        )
        return logits, spans
