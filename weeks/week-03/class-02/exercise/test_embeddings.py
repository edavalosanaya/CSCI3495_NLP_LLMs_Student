"""Tests for W3C2 embeddings.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-03/class-02/exercise/test_embeddings.py -k step2 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  EMB_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "embeddings.py"
    if os.environ.get("EMB_FROM") == "solution"
    else _HERE / "embeddings.py"
)
_spec = importlib.util.spec_from_file_location("emb_under_test", _SRC)
em = importlib.util.module_from_spec(_spec)
sys.modules["emb_under_test"] = em
_spec.loader.exec_module(em)


def test_step1_cosine_values():
    assert em.cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert em.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert em.cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_step1_cosine_ignores_magnitude():
    assert em.cosine(np.array([1.0, 1.0]), np.array([3.0, 3.0])) == pytest.approx(1.0)


def test_step2_nearest_excludes_self_and_finds_semantic_neighbor():
    res = em.nearest("cat", em.EMB, k=3)
    words = [w for w, _ in res]
    assert "cat" not in words
    # cat's closest neighbors are kitten and dog.
    assert "kitten" in words and "dog" in words


def test_step2_nearest_sorted_descending():
    res = em.nearest("king", em.EMB, k=4)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
    assert res[0][0] == "prince"  # closest to king in this toy space


def test_step3_analogy_king_queen():
    res = em.analogy("man", "king", "woman", em.EMB, k=1)
    assert res[0][0] == "queen"


def test_step3_analogy_excludes_inputs():
    res = em.analogy("man", "king", "woman", em.EMB, k=5)
    words = [w for w, _ in res]
    assert "man" not in words and "king" not in words and "woman" not in words


def test_step4_bias_direction_signs():
    # Along (woman - man): aunt should lean +, uncle should lean -.
    aunt = em.bias_score("aunt", "woman", "man", em.EMB)
    uncle = em.bias_score("uncle", "woman", "man", em.EMB)
    assert aunt > 0 > uncle


def test_step4_bias_occupation_pattern():
    # The toy space encodes a (deliberately illustrative) stereotype:
    # 'nurse' leans toward woman, 'engineer' leans toward man.
    nurse = em.bias_score("nurse", "woman", "man", em.EMB)
    engineer = em.bias_score("engineer", "woman", "man", em.EMB)
    assert nurse > engineer
