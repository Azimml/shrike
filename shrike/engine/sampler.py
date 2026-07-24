"""Batched sampling: greedy / temperature / top-p, mixed per-row params."""

from __future__ import annotations

import torch

from shrike.engine.request import Request


@torch.inference_mode()
def sample(logits: torch.Tensor, reqs: list[Request]) -> list[int]:
    if not reqs:
        return []
    temps = torch.tensor([r.sampling.temperature for r in reqs], device=logits.device)
    greedy = temps == 0.0
    out = torch.empty(len(reqs), dtype=torch.long, device=logits.device)
    out[greedy] = logits[greedy].argmax(-1)

    if (~greedy).any():
        idx = (~greedy).nonzero(as_tuple=True)[0]
        scaled = logits[idx].float() / temps[idx, None]
        probs = torch.softmax(scaled, dim=-1)
        # top-p: mask tail of the sorted cumulative distribution
        top_ps = torch.tensor([reqs[i].sampling.top_p for i in idx.tolist()], device=logits.device)
        sorted_probs, sorted_idx = probs.sort(-1, descending=True)
        cum = sorted_probs.cumsum(-1)
        keep = (cum - sorted_probs) < top_ps[:, None]  # always keep top-1
        sorted_probs = sorted_probs * keep
        choice = torch.multinomial(sorted_probs, 1)
        out[idx] = sorted_idx.gather(-1, choice).squeeze(-1)
    return out.tolist()
