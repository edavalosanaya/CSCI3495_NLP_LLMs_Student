"""Tests for W1C2 text_tools.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step2 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  TEXT_TOOLS_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "text_tools.py"
    if os.environ.get("TEXT_TOOLS_FROM") == "solution"
    else _HERE / "text_tools.py"
)
_spec = importlib.util.spec_from_file_location("text_tools_under_test", _SRC)
tt = importlib.util.module_from_spec(_spec)
sys.modules["text_tools_under_test"] = tt
_spec.loader.exec_module(tt)


def test_step1_normalize():
    assert tt.normalize("  Hello   WORLD\n") == "hello world"


def test_step2_tokenize_basic():
    assert tt.tokenize("Hello, world!") == ["hello", ",", "world", "!"]


def test_step2_tokenize_count():
    assert len(tt.tokenize("NLP is fun.")) == 4  # nlp is fun .


def test_step3_extract_emails():
    out = tt.extract("Reach me at a.b+x@mail.co or bob@x.io")
    assert out["emails"] == ["a.b+x@mail.co", "bob@x.io"]


def test_step3_extract_urls_and_mentions():
    out = tt.extract("See https://x.io/p?q=1 from @alice and @bob_99")
    assert out["urls"] == ["https://x.io/p?q=1"]
    assert out["mentions"] == ["alice", "bob_99"]


@pytest.mark.parametrize(
    "a,b,d",
    [
        ("intention", "execution", 5),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("same", "same", 0),
    ],
)
def test_step4_edit_distance(a, b, d):
    assert tt.edit_distance(a, b) == d
