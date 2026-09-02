"""Tests for W2C2 sentiment. Check one step at a time:  pytest -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  SENTIMENT_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "sentiment.py"
    if os.environ.get("SENTIMENT_FROM") == "solution"
    else _HERE / "sentiment.py"
)
_spec = importlib.util.spec_from_file_location("sentiment_under_test", _SRC)
sm = importlib.util.module_from_spec(_spec)
sys.modules["sentiment_under_test"] = sm
_spec.loader.exec_module(sm)


DOCS = [d for d, _ in sm.TRAIN]
LABELS = [y for _, y in sm.TRAIN]


def test_step1_log_prior_is_smoothed():
    # 5 of 10 docs, 2 classes -> (5+1)/(10+2)
    assert sm.log_prior(5, 10, 2) == pytest.approx(math.log(6 / 12))


def test_step1_priors_sum_to_one():
    total = sum(math.exp(sm.log_prior(n, 10, 2)) for n in (5, 5))
    assert total == pytest.approx(1.0)


def test_step2_log_likelihood_is_smoothed():
    # word seen 3 times in a class of 30 tokens, vocab 40 -> (3+1)/(30+40)
    assert sm.log_likelihood(3, 30, 40) == pytest.approx(math.log(4 / 70))


def test_step2_unseen_word_is_not_zero():
    # count 0 must still give a real (negative, finite) log-probability
    lp = sm.log_likelihood(0, 30, 40)
    assert lp == pytest.approx(math.log(1 / 70))
    assert math.isfinite(lp)


def test_step3_score_returns_one_logprob_per_class():
    m = sm.train_nb(DOCS, LABELS)
    s = sm.score(m, sm.tokenize("great story"))
    assert set(s.keys()) == set(m["classes"])
    assert all(v < 0 for v in s.values())  # log-probs are negative


def test_step3_unknown_words_are_skipped():
    m = sm.train_nb(DOCS, LABELS)
    plain = sm.score(m, sm.tokenize("great story"))
    noisy = sm.score(m, sm.tokenize("zzzqqq great story"))
    assert noisy == pytest.approx(plain)


def test_step4_predicts_clear_examples():
    m = sm.train_nb(DOCS, LABELS)
    assert sm.predict(m, sm.tokenize("wonderful brilliant great loved")) == "pos"
    assert sm.predict(m, sm.tokenize("terrible boring dreadful awful")) == "neg"


def test_step4_test_set_f1_is_perfect():
    m = sm.train_nb(DOCS, LABELS)
    gold = [y for _, y in sm.TEST]
    pred = [sm.predict(m, sm.tokenize(t)) for t, _ in sm.TEST]
    assert sm.prf(gold, pred, target="pos")["f1"] == pytest.approx(1.0)
