"""Tests for W2C2 sentiment.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-02/class-02/exercise/test_sentiment.py -k step2 -q

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


def test_step1_priors_sum_to_one():
    m = sm.train_nb(DOCS, LABELS)
    total = sum(math.exp(m["log_prior"][c]) for c in m["classes"])
    assert total == pytest.approx(1.0)


def test_step3_predicts_clear_examples():
    m = sm.train_nb(DOCS, LABELS)
    assert sm.predict(m, sm.tokenize("wonderful brilliant great loved")) == "pos"
    assert sm.predict(m, sm.tokenize("terrible boring dreadful awful")) == "neg"


def test_step2_score_returns_logprobs_per_class():
    m = sm.train_nb(DOCS, LABELS)
    s = sm.score(m, sm.tokenize("great story"))
    assert set(s.keys()) == set(m["classes"])
    assert all(v <= 0 for v in s.values())  # log-probs are non-positive


def test_step2_unknown_words_do_not_crash():
    m = sm.train_nb(DOCS, LABELS)
    # Words never seen in training should simply be skipped.
    assert sm.predict(m, sm.tokenize("zzzqqq wonderful great")) == "pos"


def test_step4_prf_perfect():
    gold = ["pos", "pos", "neg", "neg"]
    pred = ["pos", "pos", "neg", "neg"]
    m = sm.prf(gold, pred, target="pos")
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_step4_prf_mixed():
    # target=pos. gold pos: 3 (idx 0,1,2); pred pos: idx 0,1,3
    gold = ["pos", "pos", "pos", "neg"]
    pred = ["pos", "pos", "neg", "pos"]
    m = sm.prf(gold, pred, target="pos")
    # TP=2, FP=1, FN=1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_step4_prf_zero_denominator():
    gold = ["neg", "neg"]
    pred = ["neg", "neg"]
    m = sm.prf(gold, pred, target="pos")
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_step5_test_set_f1_is_reasonable():
    m = sm.train_nb(DOCS, LABELS)
    gold = [y for _, y in sm.TEST]
    pred = [sm.predict(m, sm.tokenize(t)) for t, _ in sm.TEST]
    f1 = sm.prf(gold, pred, target="pos")["f1"]
    assert f1 >= 0.5  # a sensible NB should beat chance on this set
