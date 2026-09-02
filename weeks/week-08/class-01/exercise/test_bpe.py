"""Tests for W8C1 bpe.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-08/class-01/exercise/test_bpe.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  BPE_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "bpe.py"
    if os.environ.get("BPE_FROM") == "solution"
    else _HERE / "bpe.py"
)
_spec = importlib.util.spec_from_file_location("bpe_under_test", _SRC)
bpe = importlib.util.module_from_spec(_spec)
sys.modules["bpe_under_test"] = bpe
_spec.loader.exec_module(bpe)


CORPUS = ["low low low low low", "lower lower", "newest newest newest", "widest"]


def test_given_build_vocab_counts():
    v = bpe.build_vocab(["low low lower"])
    assert v[("l", "o", "w", bpe.END)] == 2
    assert v[("l", "o", "w", "e", "r", bpe.END)] == 1


def test_step1_count_pairs_weighted():
    v = bpe.build_vocab(["low low"])  # ("l","o","w","</w>"): 2
    pairs = bpe.count_pairs(v)
    assert pairs[("l", "o")] == 2
    assert pairs[("o", "w")] == 2
    assert pairs[("w", bpe.END)] == 2


def test_step2_merge_pair():
    v = {("l", "o", "w", "e", "r", bpe.END): 3}
    merged = bpe.merge_pair(("e", "r"), v)
    assert ("l", "o", "w", "er", bpe.END) in merged
    assert merged[("l", "o", "w", "er", bpe.END)] == 3


def test_step3_train_picks_most_frequent_first():
    # "low" appears 5x; the most frequent pair should be merged first.
    merges = bpe.train_bpe(CORPUS, num_merges=1)
    assert len(merges) == 1
    # The single most frequent adjacent pair in the corpus is among low's chars.
    assert merges[0] in {("l", "o"), ("o", "w")}


def test_step3_train_is_deterministic():
    m1 = bpe.train_bpe(CORPUS, num_merges=8)
    m2 = bpe.train_bpe(CORPUS, num_merges=8)
    assert m1 == m2


def test_step3_train_stops_when_no_pairs():
    # A single one-character word has no pairs after the END merge resolves.
    merges = bpe.train_bpe(["a a a"], num_merges=50)
    # ("a","</w>") is the only pair; after merging once there are no pairs left.
    assert merges == [("a", bpe.END)]


def test_given_encode_known_word_collapses():
    merges = bpe.train_bpe(CORPUS, num_merges=12)
    enc = bpe.encode_word("low", merges)
    # "low" was very frequent; it should collapse to a single token "low</w>".
    assert enc == ["low" + bpe.END]


def test_given_encode_unknown_word_still_works():
    # A word never seen in training must still encode (no UNK), into >=1 pieces.
    merges = bpe.train_bpe(CORPUS, num_merges=12)
    enc = bpe.encode_word("zzz", merges)
    assert "".join(enc) == "zzz" + bpe.END
    assert len(enc) >= 1
