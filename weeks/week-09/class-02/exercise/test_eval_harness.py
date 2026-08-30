"""Tests for W9C2 eval_harness (offline scoring; no model needed).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-09/class-02/exercise/test_eval_harness.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  EVAL_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "eval_harness.py"
    if os.environ.get("EVAL_FROM") == "solution"
    else _HERE / "eval_harness.py"
)
_spec = importlib.util.spec_from_file_location("eval_under_test", _SRC)
eh = importlib.util.module_from_spec(_spec)
sys.modules["eval_under_test"] = eh
_spec.loader.exec_module(eh)


def test_step1_normalize_strips_articles_and_punct():
    assert eh.normalize_answer("The Paris.") == "paris"
    assert eh.normalize_answer("  A   CAT! ") == "cat"


def test_step2_exact_match():
    assert eh.exact_match("Paris", "the paris")
    assert not eh.exact_match("London", "Paris")


def test_step3_contains_answer_in_sentence():
    assert eh.contains_answer("The capital is Paris, in France.", "Paris")
    assert not eh.contains_answer("The capital is London.", "Paris")


def test_step3_contains_answer_word_boundary():
    # 'art' should not match inside 'Sparta'.
    assert not eh.contains_answer("They lived in Sparta.", "art")


def test_step4_accuracy_basic():
    preds = ["The answer is Paris.", "It is 4.", "London"]
    golds = ["Paris", "4", "Paris"]
    assert eh.accuracy(preds, golds) == pytest.approx(2 / 3)


def test_step4_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        eh.accuracy(["a"], ["a", "b"])


def test_step5_hallucination_flagged_when_not_abstaining():
    item = {"q": "Who won in 2087?", "gold": None, "answerable": False}
    # Confident fabricated answer -> hallucination.
    assert eh.is_hallucination("Dr. Jane Smith won it.", item) is True
    # Proper abstention -> not a hallucination.
    assert eh.is_hallucination("That hasn't happened; 2087 is in the future.", item) is False


def test_step5_answerable_items_never_flagged():
    item = {"q": "Capital of France?", "gold": "Paris", "answerable": True}
    assert eh.is_hallucination("Paris", item) is False


# ---- PART 2: LLM-as-judge with a position-bias check ----------------------

def _fair_judge(question, answer_a, answer_b):
    """A content-aware judge: prefers the answer containing the word 'scatter'.
    Because it judges by CONTENT (not slot), its verdict survives a swap."""
    a_good = "scatter" in answer_a.lower()
    b_good = "scatter" in answer_b.lower()
    if a_good and not b_good:
        return "A"
    if b_good and not a_good:
        return "B"
    return "tie"


def test_step0_biased_judge_always_picks_first_slot():
    assert eh.biased_judge("q", "good", "bad") == "A"
    assert eh.biased_judge("q", "bad", "good") == "A"


def test_step6_judge_pairwise_detects_inconsistency():
    # The biased judge picks slot A both times -> it favors a different *answer*
    # each run -> inconsistent.
    r = eh.judge_pairwise(eh.biased_judge, "q", "first", "second")
    assert r["winner_run1"] == "ans1"   # ans1 was in slot A on run1
    assert r["winner_run2"] == "ans2"   # ans2 was in slot A on run2
    assert r["consistent"] is False


def test_step6_judge_pairwise_consistent_for_fair_judge():
    # A content-aware judge picks the same *answer* regardless of order.
    q = "why blue?"
    a1 = "light scatters in the air"
    a2 = "the ocean reflects up"
    r = eh.judge_pairwise(_fair_judge, q, a1, a2)
    assert r["winner_run1"] == "ans1"
    assert r["winner_run2"] == "ans1"
    assert r["consistent"] is True


def test_step7_position_bias_rate_extremes():
    pairs = [
        ("q1", "alpha scatter", "beta"),
        ("q2", "gamma", "delta scatter"),
    ]
    # Fully position-biased judge: every verdict flips under swap -> 1.0
    assert eh.position_bias_rate(eh.biased_judge, pairs) == pytest.approx(1.0)
    # Fair content judge: nothing flips -> 0.0
    assert eh.position_bias_rate(_fair_judge, pairs) == pytest.approx(0.0)


def test_step7_position_bias_rate_empty():
    assert eh.position_bias_rate(eh.biased_judge, []) == pytest.approx(0.0)
