"""Tests for W2C1 ngram_lm.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-02/class-01/exercise/test_ngram_lm.py -k step2 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  NGRAM_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "ngram_lm.py"
    if os.environ.get("NGRAM_FROM") == "solution"
    else _HERE / "ngram_lm.py"
)
_spec = importlib.util.spec_from_file_location("ngram_under_test", _SRC)
ng = importlib.util.module_from_spec(_spec)
sys.modules["ngram_under_test"] = ng
_spec.loader.exec_module(ng)


TRAIN = ["the cat sat", "the cat ran", "a cat sat"]


def test_given_count_bigrams():
    m = ng.count_ngrams(TRAIN, 2)
    # "the cat" appears twice; context "the" appears twice.
    assert m["ngram"][("the", "cat")] == 2
    assert m["context"][("the",)] == 2
    # Every sentence ends with </s>.
    assert m["ngram"][("sat", "</s>")] == 2


def test_step1_prob_is_smoothed():
    m = ng.count_ngrams(TRAIN, 2)
    v = len(m["vocab"])
    # Unseen pair gets nonzero probability (add-one).
    assert ng.prob(m, ("the",), "ran") > 0
    # P(cat | the) = (2 + 1) / (count(the)=2 + V)
    assert ng.prob(m, ("the",), "cat") == pytest.approx((2 + 1) / (2 + v))


def test_step1_prob_normalizes_over_vocab():
    m = ng.count_ngrams(TRAIN, 2)
    total = sum(ng.prob(m, ("the",), w) for w in m["vocab"])
    assert total == pytest.approx(1.0, abs=1e-9)


def test_given_generate_is_deterministic():
    m = ng.count_ngrams(TRAIN, 2)
    a = ng.generate(m, 2, seed=3)
    b = ng.generate(m, 2, seed=3)
    assert a == b  # same seed -> same output
    assert ng.BOS not in a and ng.EOS not in a  # padding stripped


def test_step2_perplexity_lower_for_better_fit():
    # A model trained on a corpus should be less perplexed by an in-domain
    # sentence than a unigram model that ignores context.
    uni = ng.count_ngrams(TRAIN, 1)
    bi = ng.count_ngrams(TRAIN, 2)
    held = ["the cat sat"]
    assert ng.perplexity(bi, 2, held) < ng.perplexity(uni, 1, held)
    assert ng.perplexity(bi, 2, held) > 1.0  # never perfect on smoothed model


def test_step2_perplexity_matches_manual_unigram():
    m = ng.count_ngrams(["a a a"], 1)
    pp = ng.perplexity(m, 1, ["a a a"])
    # Recompute independently from prob().
    logs = [math.log(ng.prob(m, (), w)) for w in ["a", "a", "a", ng.EOS]]
    expected = math.exp(-sum(logs) / len(logs))
    assert pp == pytest.approx(expected)
