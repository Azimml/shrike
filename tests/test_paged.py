"""Paging + continuous batching correctness: the full engine (paged KV,
chunked prefill, batched decode) must reproduce the naive single-sequence
cache's greedy outputs token-for-token.

max_tokens_per_step=48 forces prompts to be chunk-prefilled across steps
while other sequences decode, exercising the mixed-batch path.
"""

import pytest
import torch

pytestmark = pytest.mark.integration

PROMPTS = [
    "Explain the difference between a process and a thread.",
    "Explain the difference between a mutex and a semaphore in detail.",
    "Write a haiku about memory fragmentation.",
    "What is the capital of France?",
]


@pytest.fixture(scope="module")
def engine(model_dir):
    from shrike.engine.engine import LLMEngine

    return LLMEngine(model_dir, max_tokens_per_step=48, gpu_mem_util=0.8)


def naive_greedy(engine, token_ids: list[int], max_new: int) -> list[int]:
    from shrike.models.qwen import NaiveKVBackend

    backend = NaiveKVBackend(engine.model.cfg)
    n = len(token_ids)
    ids = torch.tensor(token_ids, device="cuda")
    with torch.inference_mode():
        logits = engine.model(ids, torch.arange(n, device="cuda"), backend)
        out = []
        tok = logits[-1].argmax()
        for i in range(max_new):
            out.append(tok.item())
            if out[-1] in engine.eos_ids:
                break
            logits = engine.model(tok.view(1), torch.tensor([n + i], device="cuda"), backend)
            tok = logits[-1].argmax()
    return out


def test_engine_matches_naive(engine):
    """Free-run the engine (mixed chunked-prefill/decode batches), then
    teacher-force the naive cache over each (prompt + output) sequence:
    every engine-chosen token must be the naive argmax, except where the
    naive top-2 gap is within bf16 kernel noise (exact greedy match is not
    well-defined at near-ties — see test_parity for the same phenomenon).
    A real paging bug (wrong slot, stale block, bad mask) breaks agreement
    at confident positions immediately.
    """
    import torch

    from shrike.engine.request import SamplingParams
    from shrike.models.qwen import NaiveKVBackend

    params = SamplingParams(max_new_tokens=48, temperature=0.0)
    tokenized = [
        engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True
        )["input_ids"]
        for p in PROMPTS
    ]
    got = engine.generate(tokenized, params, chat=False)
    total, agreed = 0, 0
    for prompt_ids, engine_out in zip(tokenized, got, strict=True):
        assert len(engine_out) > 0
        backend = NaiveKVBackend(engine.model.cfg)
        seq = prompt_ids + engine_out
        n = len(prompt_ids)
        with torch.inference_mode():
            logits = engine.model(
                torch.tensor(seq[:n], device="cuda"),
                torch.arange(n, device="cuda"),
                backend,
            )
            for i, tok in enumerate(engine_out):
                row = logits[-1].float()
                total += 1
                if row.argmax().item() == tok:
                    agreed += 1
                else:
                    top2 = row.topk(2).values
                    assert (top2[0] - top2[1]).item() < 0.75, (
                        f"engine token {tok} at step {i} disagrees with naive "
                        f"argmax at a confident position"
                    )
                if i < len(engine_out) - 1:
                    logits = engine.model(
                        torch.tensor([tok], device="cuda"),
                        torch.tensor([n + i], device="cuda"),
                        backend,
                    )
    assert agreed / total > 0.9, f"only {agreed}/{total} tokens agree"


def test_all_blocks_freed(engine):
    assert engine.block_manager.num_free == engine.block_manager.num_blocks
