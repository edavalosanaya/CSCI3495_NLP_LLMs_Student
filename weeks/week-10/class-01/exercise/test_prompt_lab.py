"""Tests for W10C1 prompt_lab (offline-testable parts).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-10/class-01/exercise/test_prompt_lab.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  PROMPT_LAB_FROM=solution
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "prompt_lab.py"
    if os.environ.get("PROMPT_LAB_FROM") == "solution"
    else _HERE / "prompt_lab.py"
)
_spec = importlib.util.spec_from_file_location("prompt_lab_under_test", _SRC)
pl = importlib.util.module_from_spec(_spec)
sys.modules["prompt_lab_under_test"] = pl
_spec.loader.exec_module(pl)


def _implemented(fn, *args):
    try:
        fn(*args)
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(pl.accuracy, ["a"], ["a"]),
    reason="prompt_lab not implemented yet (fill in the TODOs)",
)


def test_step1_accuracy_perfect():
    assert pl.accuracy(["positive", "negative"], ["positive", "negative"]) == 1.0


def test_step1_accuracy_partial():
    assert pl.accuracy(["positive", "positive"], ["positive", "negative"]) == 0.5


def test_step0_parse_label():
    assert pl.parse_label("Sentiment: positive") == "positive"
    assert pl.parse_label("The answer is NEGATIVE.") == "negative"
    assert pl.parse_label("It is good, not bad") == "positive"  # first signal wins


def test_step2_build_fewshot_prompt_structure():
    if not _implemented(pl.build_fewshot_prompt, "I", [("x", "positive")], "q"):
        pytest.skip("build_fewshot_prompt not implemented")
    p = pl.build_fewshot_prompt("Classify it.", [("Great!", "positive")], "Awful.")
    assert "Classify it." in p          # instruction present
    assert "Great!" in p and "positive" in p  # demonstration present
    assert "Awful." in p                # query present
    assert p.rstrip().endswith("Sentiment:")  # output cue is last


def test_step0_stub_model_is_deterministic():
    m = pl.StubModel()
    p = pl.build_zeroshot_prompt(pl.INSTRUCTION, [], "Absolutely loved every minute of it.")
    assert m.generate(p) == m.generate(p)
    assert m.generate(p) == "positive"


def test_step3_run_experiment_offline():
    if not _implemented(pl.build_fewshot_prompt, "I", [("x", "positive")], "q"):
        pytest.skip("build_fewshot_prompt not implemented")
    m = pl.StubModel()
    preds, acc = pl.run_experiment(m, pl.build_fewshot_prompt, pl.DATASET, pl.DEMOS)
    assert len(preds) == len(pl.DATASET)
    assert all(p in ("positive", "negative") for p in preds)
    assert 0.0 <= acc <= 1.0
    # The keyword stub should get the clearly-worded items mostly right.
    assert acc >= 0.6
