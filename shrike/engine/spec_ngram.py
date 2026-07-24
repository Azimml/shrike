"""Prompt-lookup (n-gram) speculative decoding — draft-model-free speculation
(Saxena '23; shipped in vLLM as the `ngram` speculator).

Idea: during decode, if the last n tokens already occurred earlier in the
sequence (common in summarization, code, extraction — anywhere output echoes
input), guess that the tokens which followed that earlier occurrence will
follow again. The guessed tokens are verified in ONE batched forward pass:
each position's logits are computed against the real context, so accepted
tokens are exactly what greedy decoding would have produced — speculation
never changes outputs, it only trades redundant compute for parallelism.

Zero extra VRAM (no draft model), which is the only kind of speculation a
4GB GPU has room for.
"""

from __future__ import annotations


def propose(token_ids: list[int], ngram: int = 2, k: int = 4) -> list[int]:
    """Return up to k draft tokens by matching the trailing `ngram` against
    the most recent earlier occurrence in the sequence."""
    n = len(token_ids)
    if n < ngram + 1:
        return []
    pattern = tuple(token_ids[-ngram:])
    for start in range(n - ngram - 1, -1, -1):
        if tuple(token_ids[start : start + ngram]) == pattern:
            follow = token_ids[start + ngram : start + ngram + k]
            if follow:
                return list(follow)
    return []
