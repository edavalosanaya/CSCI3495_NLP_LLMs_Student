#!/usr/bin/env python3
"""W15C1, Bias probing in word embeddings (REFERENCE SOLUTION)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import the provided toy embeddings from the exercise folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exercise"))
from embeddings import EMBEDDINGS, dim  # noqa: E402


def get_vector(word: str) -> np.ndarray:
    return np.asarray(EMBEDDINGS[word], dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def association(word: str, set_a: list[str], set_b: list[str]) -> float:
    v = get_vector(word)

    total_a = 0.0
    for w in set_a:
        total_a = total_a + cosine(v, get_vector(w))
    mean_a = total_a / len(set_a)

    total_b = 0.0
    for w in set_b:
        total_b = total_b + cosine(v, get_vector(w))
    mean_b = total_b / len(set_b)

    # Averaging each side is what lets the two sets be different sizes.
    return float(mean_a - mean_b)


def effect(targets_x: list[str], targets_y: list[str],
           attr_a: list[str], attr_b: list[str]) -> float:
    # The same subtraction as association(), one level up.
    total_x = 0.0
    for w in targets_x:
        total_x = total_x + association(w, attr_a, attr_b)
    mean_x = total_x / len(targets_x)

    total_y = 0.0
    for w in targets_y:
        total_y = total_y + association(w, attr_a, attr_b)
    mean_y = total_y / len(targets_y)

    return float(mean_x - mean_y)


def report(targets_x, targets_y, attr_a, attr_b, labels=None) -> str:
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
    X = ["engineer", "programmer", "scientist"]
    Y = ["nurse", "teacher", "homemaker"]
    A = ["man", "he", "male"]
    B = ["woman", "she", "female"]
    print(report(X, Y, A, B, labels=("career", "care", "male", "female")))
