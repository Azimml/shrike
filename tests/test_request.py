"""Unit tests for Request lifecycle bookkeeping (pure Python, no model/GPU).

These pin down the invariants the scheduler and engine rely on: prompt-length
capture at construction, the generated/computed accounting, and prefill_done.
"""

from shrike.engine.request import Request, SamplingParams, Status


def _req(tokens: list[int], **sp) -> Request:
    return Request(token_ids=list(tokens), sampling=SamplingParams(**sp))


def test_prompt_length_frozen_at_construction():
    req = _req([1, 2, 3])
    assert req.num_prompt_tokens == 3
    # generated tokens are appended in place; num_prompt_tokens must not move
    req.token_ids.extend([4, 5])
    assert req.num_prompt_tokens == 3
    assert req.num_tokens == 5
    assert req.num_generated == 2


def test_defaults():
    req = _req([7])
    assert req.status is Status.WAITING
    assert req.num_computed_tokens == 0
    assert req.block_table == []
    assert req.finish_reason is None
    assert req.spec_len == 0


def test_prefill_done_tracks_computed_tokens():
    req = _req([1, 2, 3, 4])
    assert not req.prefill_done
    req.num_computed_tokens = 3
    assert not req.prefill_done  # still one prompt token to compute
    req.num_computed_tokens = 4
    assert req.prefill_done
    # once decoding, computed keeps climbing and prefill stays done
    req.token_ids.append(99)
    req.num_computed_tokens = 5
    assert req.prefill_done


def test_req_ids_are_unique_and_monotonic():
    a, b, c = _req([1]), _req([1]), _req([1])
    assert a.req_id < b.req_id < c.req_id


def test_sampling_defaults_are_greedy():
    sp = SamplingParams()
    assert sp.temperature == 0.0  # 0 => greedy
    assert sp.top_p == 1.0
    assert sp.ignore_eos is False
