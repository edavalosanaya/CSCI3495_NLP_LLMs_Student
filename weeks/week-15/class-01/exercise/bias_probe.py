#!/usr/bin/env python3
"""W15C1, Bias probing in word embeddings (STARTER).

Offline, deterministic, no network. We measure how strongly *target* words
(e.g., career vs. family) associate with *attribute* words (e.g., male vs.
female terms) using cosine similarity, a small, transparent version of the
WEAT test (Caliskan et al., 2017) that surfaces stereotype associations encoded
in embeddings.

We ship a tiny, *provided* set of toy embeddings in `embeddings.py` so the demo
runs anywhere with no downloads. The point is the *method*, not the model.

Work through the lab in `README.md`. Each STEP has its own check
(python -m pytest ... -k step1 -q). Or run everything:
    python -m pytest weeks/week-15/class-01/exercise/test_bias_probe.py -q
"""
from __future__ import annotations

import numpy as np

from embeddings import EMBEDDINGS, dim  # provided toy vectors


def get_vector(word: str) -> np.ndarray:
    """Return the embedding for `word`, or raise KeyError if unknown."""
    return np.asarray(EMBEDDINGS[word], dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    STEP 1 (check with: pytest -k step1): implement
    cos(a, b) = (a . b) / (||a|| * ||b||).
    Guard against zero-norm vectors (return 0.0 if either norm is 0).
    """
    raise NotImplementedError("Implement cosine()")


def association(word: str, set_a: list[str], set_b: list[str]) -> float:
    """Mean-cosine association of `word` toward set A vs. set B.

    s(word, A, B) = mean_{a in A} cos(word, a) - mean_{b in B} cos(word, b)

    Positive => `word` is, on average, closer to set A than to set B.

    STEP 2 (check with: pytest -k step2): compute the two mean cosines
    and return their difference.
    """
    raise NotImplementedError("Implement association()")


def effect(targets_x: list[str], targets_y: list[str],
           attr_a: list[str], attr_b: list[str]) -> float:
    """A WEAT-style summary: do X-words lean toward A while Y-words lean toward B?

    Returns mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B).
    A large positive value indicates a stereotype-aligned association pattern.

    STEP 3 (check with: pytest -k step3): average association() over X and
    over Y; return the difference.
    """
    raise NotImplementedError("Implement effect()")


def report(targets_x, targets_y, attr_a, attr_b, labels=None) -> str:
    """Human-readable summary of the probe (no TODO; uses your functions)."""
    lx, ly, la, lb = labels or ("X", "Y", "A", "B")
    rows = []
    for w in targets_x + targets_y:
        s = association(w, attr_a, attr_b)
        rows.append(f"  {w:<12} association({la} - {lb}) = {s:+.3f}")
    e = effect(targets_x, targets_y, attr_a, attr_b)
    body = "\n".join(rows)
    return (
        f"Bias probe (toy embeddings, dim={dim()})\n"
        f"  targets {lx}: {targets_x}\n  targets {ly}: {targets_y}\n"
        f"  attributes {la}: {attr_a}\n  attributes {lb}: {attr_b}\n"
        f"{body}\n"
        f"  EFFECT ({lx} leans {la} & {ly} leans {lb}) = {e:+.3f}\n"
    )


if __name__ == "__main__":
    # A classic probe shape: career/family vs. male/female terms.
    X = ["engineer", "programmer", "scientist"]   # "career"-coded targets
    Y = ["nurse", "teacher", "homemaker"]         # "family"/care-coded targets
    A = ["man", "he", "male"]
    B = ["woman", "she", "female"]
    print(report(X, Y, A, B, labels=("career", "care", "male", "female")))
    print("Reflect: the toy vectors were *constructed* to show this pattern.")
    print("Real embeddings (Week 3) show similar, well-documented biases.")
