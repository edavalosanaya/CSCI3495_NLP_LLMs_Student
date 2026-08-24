"""Tests for W7C1 decoding (pure Python, no model needed).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-07/class-01/exercise/test_decoding.py -k step1 -q

Runs against the student's exercise file by default. Set DECODING_FROM=solution
to test the reference solution (used by the course sweep).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "decoding.py"
    if os.environ.get("DECODING_FROM") == "solution"
    else _HERE / "decoding.py"
)
_spec = importlib.util.spec_from_file_location("decoding_under_test", _SRC)
dec = importlib.util.module_from_spec(_spec)
sys.modules["decoding_under_test"] = dec
_spec.loader.exec_module(dec)


def _implemented():
    try:
        dec.greedy({"a": 1.0})
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(not _implemented(), reason="decoding not implemented yet")

LOGITS = {"a": 2.0, "b": 1.0, "c": 0.0, "d": -1.0}


def test_step1_temperature_normalizes():
    dist = dec.apply_temperature(LOGITS, 1.0)
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-6)


def test_step1_low_temperature_is_sharper():
    sharp = dec.apply_temperature(LOGITS, 0.5)
    flat = dec.apply_temperature(LOGITS, 2.0)
    # Lower temperature => MORE mass on the top token than higher temperature.
    assert sharp["a"] > flat["a"]


def test_step1_near_zero_temperature_is_greedy():
    dist = dec.apply_temperature(LOGITS, 1e-9)
    assert math.isclose(dist["a"], 1.0, rel_tol=1e-6)
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-6)


def test_step2_greedy_picks_argmax():
    dist = dec.apply_temperature(LOGITS, 1.0)
    assert dec.greedy(dist) == "a"


def test_step3_top_k_keeps_k_and_normalizes():
    dist = dec.apply_temperature(LOGITS, 1.0)
    filt = dec.top_k_filter(dist, 2)
    assert len(filt) == 2
    assert set(filt) == {"a", "b"}
    assert math.isclose(sum(filt.values()), 1.0, rel_tol=1e-6)


def test_step4_top_p_keeps_nucleus():
    dist = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    filt = dec.top_p_filter(dist, 0.7)
    # smallest top set summing to >= 0.7 is {a, b} (0.5 + 0.3 = 0.8)
    assert set(filt) == {"a", "b"}
    assert math.isclose(sum(filt.values()), 1.0, rel_tol=1e-6)


def test_step4_top_p_always_keeps_at_least_one():
    dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    filt = dec.top_p_filter(dist, 0.0)
    assert len(filt) >= 1


def test_step5_sample_is_deterministic_with_seed():
    dist = dec.apply_temperature(LOGITS, 1.0)
    s1 = dec.sample(dist, seed=42)
    s2 = dec.sample(dist, seed=42)
    assert s1 == s2
    assert s1 in dist


def test_step5_sample_respects_certainty():
    # All mass on one token -> always that token.
    dist = {"x": 1.0, "y": 0.0}
    assert dec.sample(dist, seed=1) == "x"
    assert dec.sample(dist, seed=99) == "x"
