"""Qwen2 architecture in pure PyTorch (RoPE + GQA + RMSNorm + SwiGLU).

Loads HF safetensors directly. `transformers` is not imported here — the only
external inputs are config.json and model.safetensors from the HF snapshot.

The attention backend (naive contiguous KV cache vs paged KV cache) is
injected per forward pass, so the same transformer code serves the parity
test, the naive baseline, and the paged batched engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class QwenConfig:
    def __init__(self, model_dir: str | Path):
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"no config.json under {model_dir!r}; download a snapshot with "
                "`python scripts/download_model.py` or set SHRIKE_MODEL_DIR to an "
                "existing model directory"
            )
        raw = json.loads(config_path.read_text())
        self.hidden_size: int = raw["hidden_size"]
        self.intermediate_size: int = raw["intermediate_size"]
        self.num_layers: int = raw["num_hidden_layers"]
        self.num_heads: int = raw["num_attention_heads"]
        self.num_kv_heads: int = raw["num_key_value_heads"]
        self.head_dim: int = raw.get("head_dim") or self.hidden_size // self.num_heads
        self.vocab_size: int = raw["vocab_size"]
        self.rms_norm_eps: float = raw["rms_norm_eps"]
        self.rope_theta: float = raw["rope_theta"]
        self.tie_word_embeddings: bool = raw.get("tie_word_embeddings", False)
        self.max_position_embeddings: int = raw["max_position_embeddings"]
        self.eos_token_ids: list[int] = (
            raw["eos_token_id"]
            if isinstance(raw.get("eos_token_id"), list)
            else [raw["eos_token_id"]]
        )
        self.dtype = getattr(torch, raw.get("torch_dtype", "bfloat16"))


class AttentionBackend(Protocol):
    """Owns the KV cache and computes attention for one layer.

    q: [N, H, D], k/v: [N, H_kv, D] for the N new tokens in this step.
    Returns attention output [N, H, D].
    """

    def run(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, device: torch.device | str):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [N] -> cos/sin [N, head_dim]
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q: [N, H, D], k: [N, H_kv, D], cos/sin: [N, D]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_out = (q.float() * cos + rotate_half(q.float()) * sin).to(q.dtype)
    k_out = (k.float() * cos + rotate_half(k.float()) * sin).to(k.dtype)
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, cfg: QwenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backend: AttentionBackend
    ):
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        out = backend.run(self.layer_idx, q, k, v)
        return self.o_proj(out.reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: QwenConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, backend):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, backend)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class QwenModel(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, cos, sin, backend):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, backend)
        return self.norm(x)


class QwenForCausalLM(nn.Module):
    """Flat-token interface: input_ids [N], positions [N] -> logits [N, vocab].

    Batching (which tokens belong to which sequence) is entirely the attention
    backend's concern; the dense layers are position-independent.
    """

    def __init__(self, cfg: QwenConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.model = QwenModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        backend: AttentionBackend,
        out_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cos, sin = self.rotary(positions)
        hidden = self.model(input_ids, cos, sin, backend)
        if out_rows is not None:  # skip lm_head for rows nobody samples from
            hidden = hidden[out_rows]
        return self.lm_head(hidden)

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cuda") -> QwenForCausalLM:
        cfg = QwenConfig(model_dir)
        dev = torch.device(device)
        with torch.device("meta"):
            model = cls(cfg, device=dev)
        model.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, dev)  # rebuild off-meta

        shards = sorted(Path(model_dir).glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"no *.safetensors weight shards under {model_dir!r}; the snapshot "
                "looks incomplete — re-run `python scripts/download_model.py`"
            )
        state = {}
        for shard in shards:
            state.update(load_file(shard))
        if cfg.tie_word_embeddings and "lm_head.weight" not in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        state = {k: v.to(cfg.dtype) for k, v in state.items()}
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(dev).eval()


class NaiveKVBackend:
    """Contiguous per-layer KV cache for a single sequence — the 'rung 2'
    textbook cache and the reference implementation for correctness tests."""

    def __init__(self, cfg: QwenConfig):
        self.k: list[torch.Tensor | None] = [None] * cfg.num_layers
        self.v: list[torch.Tensor | None] = [None] * cfg.num_layers
        self.num_kv_groups = cfg.num_heads // cfg.num_kv_heads

    def run(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        prev_k, prev_v = self.k[layer_idx], self.v[layer_idx]
        if prev_k is None or prev_v is None:
            keys, values = k, v
        else:
            keys = torch.cat([prev_k, k], dim=0)
            values = torch.cat([prev_v, v], dim=0)
        self.k[layer_idx], self.v[layer_idx] = keys, values

        n, c = q.shape[0], keys.shape[0]
        # [H, N, D] x [H_kv, C, D]; SDPA wants batch dims in front
        q_ = q.transpose(0, 1).unsqueeze(0)
        k_ = keys.transpose(0, 1).unsqueeze(0)
        v_ = values.transpose(0, 1).unsqueeze(0)
        if n == c:
            out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True, enable_gqa=True)
        else:
            # decode/chunk step: causal mask aligned to the end of the context
            mask = torch.ones(n, c, dtype=torch.bool, device=q.device).tril(diagonal=c - n)
            out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, enable_gqa=True)
        return out.squeeze(0).transpose(0, 1)
