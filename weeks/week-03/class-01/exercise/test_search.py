"""Tests for W3C1 search.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-03/class-01/exercise/test_search.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  SEARCH_FROM=solution  (used by the course test sweep).
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
    _HERE.parent / "solutions" / "search.py"
    if os.environ.get("SEARCH_FROM") == "solution"
    else _HERE / "search.py"
)
_spec = importlib.util.spec_from_file_location("search_under_test", _SRC)
se = importlib.util.module_from_spec(_spec)
sys.modules["search_under_test"] = se
_spec.loader.exec_module(se)


def test_given_idf_ranks_rare_above_common():
    idx = se.build_index(["the cat", "the dog", "the bird"])
    # "the" is in every doc -> idf 0; "cat" in one -> higher idf.
    assert idx["idf"]["the"] == pytest.approx(0.0)
    assert idx["idf"]["cat"] > idx["idf"]["the"]


def test_step1_tfidf_drops_zero_idf_terms():
    idx = se.build_index(["the cat", "the dog"])
    vec = se.tfidf_vector(idx, ["the", "the", "cat"])
    assert "the" not in vec  # idf 0 -> weight 0 -> dropped
    assert vec["cat"] > 0


def test_step2_cosine_basics():
    assert se.cosine({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert se.cosine({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)
    assert se.cosine({}, {"a": 1.0}) == 0.0  # zero norm -> 0


def test_step2_cosine_ignores_magnitude():
    # Same direction, different magnitudes -> cosine 1.
    assert se.cosine({"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 6.0}) == pytest.approx(1.0)


def test_given_search_finds_relevant_doc_first():
    idx = se.build_index(se.DOCS)
    top = se.search(idx, "hot dog mustard", k=1)
    assert top[0][0] == 3  # the grilled-hot-dog document


def test_given_search_returns_k_results_sorted():
    idx = se.build_index(se.DOCS)
    results = se.search(idx, "team championship game", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_given_search_ties_break_by_doc_id():
    # A query term absent from all docs -> all scores 0 -> ascending doc_id.
    idx = se.build_index(se.DOCS)
    results = se.search(idx, "zzzznonexistent", k=3)
    assert [d for d, _ in results] == [0, 1, 2]
