"""Tests for W10C2 cot_lab (offline-testable parts).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-10/class-02/exercise/test_cot_lab.py -k step1 -q

Set COT_LAB_FROM=solution to test the reference solution.
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "cot_lab.py"
    if os.environ.get("COT_LAB_FROM") == "solution"
    else _HERE / "cot_lab.py"
)
_spec = importlib.util.spec_from_file_location("cot_lab_under_test", _SRC)
cl = importlib.util.module_from_spec(_spec)
sys.modules["cot_lab_under_test"] = cl
_spec.loader.exec_module(cl)


def _implemented(fn, *args):
    try:
        fn(*args)
        return True
    except NotImplementedError:
        return False


def test_given_extract_answer_last_int():
    assert cl.extract_answer("Start with 23, then 6, the answer is 18.") == 18
    assert cl.extract_answer("Answer: -4") == -4
    assert cl.extract_answer("no numbers here") is None


def test_step1_majority_vote():
    if not _implemented(cl.majority_vote, [1]):
        pytest.skip("majority_vote not implemented")
    assert cl.majority_vote([18, 18, 19]) == 18
    assert cl.majority_vote([2, 3, 2, 3]) == 2  # tie -> smallest
    assert cl.majority_vote([None, 5, 5]) == 5
    assert cl.majority_vote([None, None]) is None


def test_step2_stub_cot_beats_direct():
    """The offline stub is built so CoT outscores direct, the lesson of the day."""
    if not _implemented(cl.evaluate, cl.StubModel(), cl.direct_prompt, cl.DATASET[:1]):
        pytest.skip("evaluate not implemented")
    m = cl.StubModel()
    direct = cl.evaluate(m, cl.direct_prompt, cl.DATASET)
    cot = cl.evaluate(m, cl.cot_prompt, cl.DATASET)
    assert cot == 1.0          # reasoning reveals the right answer
    assert direct < cot        # direct guessing does worse


def test_given_stub_is_deterministic():
    m = cl.StubModel()
    p = cl.cot_prompt(cl.DATASET[0][0])
    assert m.generate(p) == m.generate(p)
