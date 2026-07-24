"""Unit tests for prompt-lookup (n-gram) speculative drafting — pure Python,
no model or GPU."""

from shrike.engine.spec_ngram import propose


def test_no_proposal_when_sequence_too_short():
    # need at least ngram + 1 tokens to have both a pattern and something after
    assert propose([1, 2], ngram=2, k=4) == []
    assert propose([], ngram=2) == []


def test_matches_most_recent_earlier_occurrence():
    # trailing bigram (1, 2) last occurred at index 0, followed by 3, 4, 5
    seq = [1, 2, 3, 4, 5, 9, 9, 1, 2]
    assert propose(seq, ngram=2, k=3) == [3, 4, 5]


def test_k_caps_the_draft_length():
    seq = [1, 2, 3, 4, 5, 6, 7, 1, 2]
    assert propose(seq, ngram=2, k=2) == [3, 4]


def test_picks_latest_match_not_earliest():
    # bigram (7, 8) occurs twice; the most recent one (index 4) is followed by
    # 99, which is what should be drafted
    seq = [7, 8, 1, 2, 7, 8, 99, 3, 4, 7, 8]
    assert propose(seq, ngram=2, k=1) == [99]


def test_no_match_returns_empty():
    seq = [1, 2, 3, 4, 5, 6, 7, 8]  # trailing (7, 8) never occurred before
    assert propose(seq, ngram=2, k=4) == []


def test_trailing_pattern_at_end_of_history_yields_nothing():
    # the only earlier occurrence of (2, 3) has nothing following it before the
    # trailing copy, so there is no continuation to draft
    seq = [1, 2, 3, 2, 3]
    # earlier (2, 3) at index 1 is followed by 2 -> a valid one-token draft
    assert propose(seq, ngram=2, k=4) == [2, 3]


def test_larger_ngram():
    seq = [5, 6, 7, 8, 1, 1, 1, 5, 6, 7]
    # trailing trigram (5, 6, 7) matches index 0, followed by 8
    assert propose(seq, ngram=3, k=2) == [8, 1]
