"""Hugging Face throughput baselines for the benchmark ladder."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.prompts import PROMPTS
from bench.workload import target_lengths
from shrike.config import DEFAULT_MODEL_DIR

MODEL_DIR = DEFAULT_MODEL_DIR
RESULTS_PATH = Path("bench/results/results.json")
RUNG_NAMES = {
    1: "no_kv_cache",
    2: "kv_cache",
    3: "static_batching",
}


def _render_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def _tokenize_individually(tokenizer: Any, prompts: list[str]) -> list[torch.Tensor]:
    rendered = _render_prompts(tokenizer, prompts)
    return [
        tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        for text in rendered
    ]


def _run_no_kv_cache(model: Any, prompt_ids: list[torch.Tensor], lengths: list[int]) -> float:
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode():
        for input_ids, num_new in zip(prompt_ids, lengths, strict=True):
            full_ids = input_ids
            for _ in range(num_new):
                logits = model(full_ids, use_cache=False).logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                full_ids = torch.cat((full_ids, next_token), dim=1)
    torch.cuda.synchronize()
    return time.perf_counter() - started_at


def _run_kv_cache(model: Any, prompt_ids: list[torch.Tensor], lengths: list[int]) -> float:
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode():
        for input_ids, num_new in zip(prompt_ids, lengths, strict=True):
            outputs = model(input_ids, use_cache=True)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            for _ in range(num_new - 1):
                outputs = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
    torch.cuda.synchronize()
    return time.perf_counter() - started_at


def _run_static_batch(
    model: Any, tokenizer: Any, prompts: list[str], lengths: list[int]
) -> tuple[float, int, int, int]:
    """The batch must run until its LONGEST member finishes (that is the
    static-batching pathology): every row decodes max(lengths) tokens, but
    only each row's own target length counts as useful output.

    Returns (seconds, batch_size, useful_tokens, raw_tokens).
    """
    rendered = _render_prompts(tokenizer, prompts)
    batch_size = len(rendered)

    while batch_size >= 1:
        try:
            encoded = tokenizer(
                rendered[:batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to("cuda")
            batch_max = max(lengths[:batch_size])
            torch.cuda.synchronize()
            started_at = time.perf_counter()
            with torch.inference_mode():
                model.generate(
                    **encoded,
                    do_sample=False,
                    min_new_tokens=batch_max,
                    max_new_tokens=batch_max,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started_at
            useful = sum(lengths[:batch_size])
            return seconds, batch_size, useful, batch_size * batch_max
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size //= 2
            gc.collect()
            torch.cuda.empty_cache()
            print(f"CUDA OOM; retrying static batching with {batch_size} prompts")

    raise RuntimeError("Unable to find a static batch size that fits on the GPU")


def _write_result(name: str, result: dict[str, int | float]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as result_file:
            results = json.load(result_file)
        if not isinstance(results, dict):
            raise ValueError(f"Expected a JSON object in {RESULTS_PATH}")
    else:
        results = {}
    results[name] = result
    with RESULTS_PATH.open("w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)
        result_file.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="prompts to measure (default: 8 for rung 1, 16 otherwise)",
    )
    args = parser.parse_args()
    if args.num_prompts is None:
        args.num_prompts = 8 if args.rung == 1 else 16
    if not 1 <= args.num_prompts <= len(PROMPTS):
        parser.error(f"--num-prompts must be between 1 and {len(PROMPTS)}")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The throughput baselines require a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    prompts = PROMPTS[: args.num_prompts]
    lengths = target_lengths(len(PROMPTS))[: args.num_prompts]
    raw_tokens = None
    if args.rung == 1:
        seconds = _run_no_kv_cache(model, _tokenize_individually(tokenizer, prompts), lengths)
        num_prompts, tokens = len(prompts), sum(lengths)
    elif args.rung == 2:
        seconds = _run_kv_cache(model, _tokenize_individually(tokenizer, prompts), lengths)
        num_prompts, tokens = len(prompts), sum(lengths)
    else:
        seconds, num_prompts, tokens, raw_tokens = _run_static_batch(
            model, tokenizer, prompts, lengths
        )

    result: dict[str, int | float] = {
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds,
        "num_prompts": num_prompts,
    }
    if raw_tokens is not None:
        result["raw_tokens"] = raw_tokens
        result["raw_tok_s"] = raw_tokens / seconds
    rung_name = RUNG_NAMES[args.rung]
    _write_result(rung_name, result)
    print(f"{rung_name}: {tokens} tokens in {seconds:.3f}s ({tokens / seconds:.2f} tok/s)")


if __name__ == "__main__":
    main()
