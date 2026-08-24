"""Tests for W15C1 bias_probe.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-15/class-01/exercise/test_bias_probe.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  BIAS_PROBE_FROM=solution  (used by the course test sweep).

Fully offline & deterministic: uses the provided toy embeddings.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
# Both versions import `embeddings` from the exercise dir; ensure it's importable.
sys.path.insert(0, str(_HERE))

_SRC = (
    _HERE.parent / "solutions" / "bias_probe.py"
    if os.environ.get("BIAS_PROBE_FROM") == "solution"
    else _HERE / "bias_probe.py"
)
_spec = importlib.util.spec_from_file_location("bias_probe_under_test", _SRC)
bp = importlib.util.module_from_spec(_spec)
sys.modules["bias_probe_under_test"] = bp
_spec.loader.exec_module(bp)


def _implemented():
    try:
        bp.cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(),
    reason="bias_probe not implemented yet (fill in the TODOs)",
)


# ---- cosine ----

def test_step1_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert bp.cosine(v, v) == pytest.approx(1.0)


def test_step1_cosine_orthogonal_is_zero():
    assert bp.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_step1_cosine_opposite_is_minus_one():
    assert bp.cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_step1_cosine_handles_zero_vector():
    assert bp.cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


# ---- association ----

def test_step2_association_sign_for_male_coded_word():
    # "engineer" is constructed male-leaning -> closer to male attrs than female.
    s = bp.association("engineer", ["man", "he", "male"], ["woman", "she", "female"])
    assert s > 0


def test_step2_association_sign_for_female_coded_word():
    s = bp.association("nurse", ["man", "he", "male"], ["woman", "she", "female"])
    assert s < 0


def test_step2_association_neutral_word_near_zero():
    s = bp.association("music", ["man", "he", "male"], ["woman", "she", "female"])
    assert abs(s) < 0.2


# ---- effect (WEAT-style summary) ----

def test_step3_effect_is_positive_for_stereotype_pattern():
    X = ["engineer", "programmer", "scientist"]
    Y = ["nurse", "teacher", "homemaker"]
    A = ["man", "he", "male"]
    B = ["woman", "she", "female"]
    e = bp.effect(X, Y, A, B)
    assert e > 0.3  # career-words lean male AND care-words lean female


def test_step3_effect_flips_sign_when_attributes_swapped():
    X = ["engineer", "programmer", "scientist"]
    Y = ["nurse", "teacher", "homemaker"]
    A = ["man", "he", "male"]
    B = ["woman", "she", "female"]
    e1 = bp.effect(X, Y, A, B)
    e2 = bp.effect(X, Y, B, A)  # swap attribute sets
    assert e2 == pytest.approx(-e1, abs=1e-9)


def test_step3_effect_deterministic():
    args = (["engineer"], ["nurse"], ["man"], ["woman"])
    assert bp.effect(*args) == bp.effect(*args)
