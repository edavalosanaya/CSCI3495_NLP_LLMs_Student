"""Tests for W8C2 preferences (Bradley-Terry reward model).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-08/class-02/exercise/test_preferences.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  PREFS_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "preferences.py"
    if os.environ.get("PREFS_FROM") == "solution"
    else _HERE / "preferences.py"
)
_spec = importlib.util.spec_from_file_location("prefs_under_test", _SRC)
pref = importlib.util.module_from_spec(_spec)
sys.modules["prefs_under_test"] = pref
_spec.loader.exec_module(pref)


def test_step1_sigmoid_basic():
    assert abs(pref.sigmoid(0.0) - 0.5) < 1e-9
    assert pref.sigmoid(50) > 0.999
    assert pref.sigmoid(-50) < 0.001


def test_step1_sigmoid_stable_on_large_inputs():
    # Must not overflow.
    assert 0.0 <= pref.sigmoid(1000) <= 1.0
    assert 0.0 <= pref.sigmoid(-1000) <= 1.0


def test_step2_nll_lower_for_correct_scores():
    prefs = [("A", "B")]
    good = {"A": 5.0, "B": -5.0}   # A clearly preferred -> low loss
    bad = {"A": -5.0, "B": 5.0}    # contradicts the label -> high loss
    assert pref.neg_log_likelihood(good, prefs) < pref.neg_log_likelihood(bad, prefs)


def test_step3_fit_recovers_ranking():
    scores = pref.fit_reward_model(pref.PREFERENCES, steps=800)
    ranking = sorted(scores, key=scores.get, reverse=True)
    assert ranking == ["A", "B", "C", "D"]


def test_step3_fit_reduces_loss():
    start = {x: 0.0 for x in {a for p in pref.PREFERENCES for a in p}}
    start_loss = pref.neg_log_likelihood(start, pref.PREFERENCES)
    fitted = pref.fit_reward_model(pref.PREFERENCES, steps=800)
    end_loss = pref.neg_log_likelihood(fitted, pref.PREFERENCES)
    assert end_loss < start_loss


def test_step3_scores_centered():
    scores = pref.fit_reward_model(pref.PREFERENCES, steps=500)
    assert abs(sum(scores.values())) < 1e-6
