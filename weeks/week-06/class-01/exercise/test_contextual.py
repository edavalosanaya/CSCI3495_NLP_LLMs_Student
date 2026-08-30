"""Tests for W6C1 contextual_embeddings.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-06/class-01/exercise/test_contextual.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution (used by the course test sweep), set:
    CONTEXTUAL_FROM=solution

The cosine test always runs (pure math). The model-based tests skip gracefully
if (a) the functions aren't implemented yet, or (b) the tiny model can't be
loaded (e.g. offline with no cache).
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
    _HERE.parent / "solutions" / "contextual_embeddings.py"
    if os.environ.get("CONTEXTUAL_FROM") == "solution"
    else _HERE / "contextual_embeddings.py"
)
_spec = importlib.util.spec_from_file_location("contextual_under_test", _SRC)
ce = importlib.util.module_from_spec(_spec)
sys.modules["contextual_under_test"] = ce
_spec.loader.exec_module(ce)


def _cosine_implemented() -> bool:
    try:
        ce.cosine_similarity([1.0, 0.0], [1.0, 0.0])
        return True
    except NotImplementedError:
        return False


cosine_only = pytest.mark.skipif(
    not _cosine_implemented(), reason="cosine_similarity not implemented yet"
)


@cosine_only
def test_step1_cosine_identical():
    assert math.isclose(ce.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, rel_tol=1e-6)


@cosine_only
def test_step1_cosine_orthogonal():
    assert math.isclose(ce.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


@cosine_only
def test_step1_cosine_scale_invariant():
    a = ce.cosine_similarity([1.0, 1.0], [2.0, 2.0])
    assert math.isclose(a, 1.0, rel_tol=1e-6)


def _model_ready():
    """True if the model functions are implemented AND the model loads."""
    try:
        ce.static_vector("bank")
        return True
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001  (offline / download failure)
        return False


needs_model = pytest.mark.skipif(
    not _model_ready(),
    reason="model functions not implemented or tiny model unavailable (offline)",
)

_SENT_RIVER = "I sat by the river bank and watched the water."
_SENT_MONEY = "I deposited my paycheck at the bank downtown."


@needs_model
def test_step3_static_is_context_independent():
    # Static vector for the SAME word is identical regardless of any sentence.
    v1 = ce.static_vector("bank")
    v2 = ce.static_vector("bank")
    assert math.isclose(ce.cosine_similarity(v1, v2), 1.0, rel_tol=1e-6)


@needs_model
def test_step2_contextual_differs_by_sentence():
    cv_river = ce.contextual_vector(_SENT_RIVER, "bank")
    cv_money = ce.contextual_vector(_SENT_MONEY, "bank")
    sim = ce.cosine_similarity(cv_river, cv_money)
    # Same word, different senses -> contextual vectors are NOT identical.
    assert sim < 0.999, f"contextual vectors should differ, got cos={sim:.4f}"


@needs_model
def test_step4_contextual_less_similar_than_static():
    sv = ce.cosine_similarity(ce.static_vector("bank"), ce.static_vector("bank"))
    cv = ce.cosine_similarity(
        ce.contextual_vector(_SENT_RIVER, "bank"),
        ce.contextual_vector(_SENT_MONEY, "bank"),
    )
    # The whole point: context pulls the two 'bank' vectors apart.
    assert cv < sv
