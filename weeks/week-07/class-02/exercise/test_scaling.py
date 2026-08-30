"""Tests for W7C2 scaling scoring core (pure Python, no model needed).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-07/class-02/exercise/test_scaling.py -k step1 -q

Runs against the student's exercise file by default. Set SCALING_FROM=solution
to test the reference solution (used by the course sweep).
"""
import importlib.util
import inspect
import math
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "scaling.py"
    if os.environ.get("SCALING_FROM") == "solution"
    else _HERE / "scaling.py"
)
_spec = importlib.util.spec_from_file_location("scaling_under_test", _SRC)
sc = importlib.util.module_from_spec(_spec)
sys.modules["scaling_under_test"] = sc
_spec.loader.exec_module(sc)


def test_step1_normalize():
    assert sc.normalize("  Paris.  ") == "paris"
    assert sc.normalize("7!") == "7"


def test_step2_is_correct_lenient_substring():
    assert sc.is_correct("The answer is 4.", "4")
    assert sc.is_correct("Paris", "paris")
    assert sc.is_correct("7 days a week", "7")
    assert not sc.is_correct("Lyon", "paris")


def test_step3_accuracy():
    targets = ["4", "paris", "7"]
    outs = ["4", "Lyon", "7 days"]
    assert math.isclose(sc.accuracy(outs, targets), 2 / 3, rel_tol=1e-6)


def test_step3_accuracy_empty():
    assert sc.accuracy([], []) == 0.0


def test_step4_scaling_trend_increasing():
    assert sc.scaling_trend({"0.5b": 0.4, "1b": 0.6, "3b": 0.8}) is True


def test_step4_scaling_trend_flat_ok():
    assert sc.scaling_trend({"a": 0.5, "b": 0.5}) is True


def test_step4_scaling_trend_decreasing_false():
    assert sc.scaling_trend({"small": 0.8, "large": 0.6}) is False


def test_step5_end_to_end_simulated():
    targets = [t["answer"] for t in sc.TASKS]
    small = ["4", "Lyon", "7", "green", "six"]
    large = ["4", "Paris.", "7 days", "Green", "7"]
    a_small = sc.accuracy(small, targets)
    a_large = sc.accuracy(large, targets)
    assert a_large > a_small
    assert sc.scaling_trend({"small": a_small, "large": a_large}) is True
