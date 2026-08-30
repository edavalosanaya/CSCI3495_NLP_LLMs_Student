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
    mean_a = np.mean([cosine(v, get_vector(w)) for w in set_a])
    mean_b = np.mean([cosine(v, get_vector(w)) for w in set_b])
    return float(mean_a - mean_b)


def effect(targets_x: list[str], targets_y: list[str],
           attr_a: list[str], attr_b: list[str]) -> float:
    mean_x = np.mean([association(w, attr_a, attr_b) for w in targets_x])
    mean_y = np.mean([association(w, attr_a, attr_b) for w in targets_y])
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
