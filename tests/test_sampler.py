"""Unit tests for the batched sampler on synthetic logits (CPU, no model)."""

import torch

from shrike.engine.request import Request, SamplingParams
from shrike.engine.sampler import sample


def _req(temperature: float = 0.0, top_p: float = 1.0) -> Request:
    return Request(
        token_ids=[1],
        sampling=SamplingParams(temperature=temperature, top_p=top_p),
    )


def test_empty_batch():
    assert sample(torch.zeros(0, 5), []) == []


def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.0, 0.0]])
    out = sample(logits, [_req(0.0), _req(0.0)])
    assert out == [1, 0]


def test_temperature_sampling_respects_seed():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    a = sample(logits.clone(), [_req(temperature=1.0)])
    torch.manual_seed(0)
    b = sample(logits.clone(), [_req(temperature=1.0)])
    assert a == b  # deterministic under a fixed seed
    assert 0 <= a[0] < 4


def test_top_p_restricts_to_dominant_token():
    # one token has almost all the mass; nucleus sampling with a small top_p
    # must always select it
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    for seed in range(20):
        torch.manual_seed(seed)
        assert sample(logits.clone(), [_req(temperature=1.0, top_p=0.5)]) == [0]


def test_mixed_greedy_and_sampled_rows():
    # row 0 greedy, row 1 sampled; greedy row must be exact, sampled row valid
    logits = torch.tensor([[0.0, 0.0, 7.0], [50.0, 0.0, 0.0]])
    torch.manual_seed(1)
    out = sample(logits, [_req(0.0), _req(temperature=1.0, top_p=0.9)])
    assert out[0] == 2
    assert out[1] == 0  # dominated by the huge logit even when sampled
