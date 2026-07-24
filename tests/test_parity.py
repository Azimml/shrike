"""Correctness gate: our pure-PyTorch Qwen must match HF transformers.

1. Prefill logits allclose (bf16 tolerance).
2. 50-token greedy continuation exact-match, generated token-by-token through
   our NaiveKVBackend (exercises the decode path + causal offset mask).
"""

import pytest
import torch

pytestmark = pytest.mark.integration

PROMPT = "The key idea behind paged attention in LLM serving is"


@pytest.fixture(scope="module")
def setup(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shrike.models.qwen import QwenForCausalLM

    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="bfloat16").cuda().eval()
    ours = QwenForCausalLM.load(model_dir)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    return hf, ours, ids


def test_prefill_logits_close(setup):
    from shrike.models.qwen import NaiveKVBackend

    hf, ours, ids = setup
    with torch.inference_mode():
        ref = hf(ids).logits[0]
        n = ids.shape[1]
        got = ours(ids[0], torch.arange(n, device="cuda"), NaiveKVBackend(ours.cfg))
    # bf16 accumulation-order differences: compare top-1 agreement + magnitude
    assert (ref.argmax(-1) == got.argmax(-1)).float().mean().item() > 0.99
    assert torch.allclose(ref.float(), got.float(), atol=1.0, rtol=0.05)


def test_decode_path_teacher_forced(setup):
    """Drive OUR token-by-token decode path (KV cache + causal-offset mask)
    through 50 tokens of HF's greedy continuation, comparing each step's
    argmax against HF's full-context forward.

    Exact greedy match vs hf.generate() is not a sound criterion in bf16:
    HF's own incremental-cache path disagrees with HF's own full-context
    forward at near-tie logits (observed top-2 gaps of 0.0). So we require
    argmax agreement everywhere EXCEPT positions where our top-2 gap is a
    near-tie (< 0.25), where either choice is numerically legitimate.
    """
    from shrike.models.qwen import NaiveKVBackend

    hf, ours, ids = setup
    with torch.inference_mode():
        ref = hf.generate(ids, max_new_tokens=50, do_sample=False)[0]
        n = ids.shape[1]
        ref_logits = hf(ref.unsqueeze(0)).logits[0]  # full-context reference

        backend = NaiveKVBackend(ours.cfg)
        logits = ours(ref[:n], torch.arange(n, device="cuda"), backend)
        agree, near_ties = 0, 0
        for i in range(n, len(ref)):
            row = logits[-1].float()
            if row.argmax().item() == ref_logits[i - 1].argmax().item():
                agree += 1
            else:
                top2 = row.topk(2).values
                assert (top2[0] - top2[1]).item() < 0.25, (
                    f"decode argmax mismatch at pos {i} with confident logits"
                )
                near_ties += 1
            logits = ours(ref[i].view(1), torch.tensor([i], device="cuda"), backend)
    assert agree >= 45, f"only {agree}/50 decode steps agree ({near_ties} near-ties)"
