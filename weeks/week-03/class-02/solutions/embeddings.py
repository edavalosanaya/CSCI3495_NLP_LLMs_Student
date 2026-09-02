#!/usr/bin/env python3
"""W3C2 reference solution, explore word embeddings."""
from __future__ import annotations

import numpy as np

# Dimensions (illustrative): 0=royal, 1=masculine, 2=feminine, 3=animal,
# 4=pet, 5=care-work, 6=technical, 7=high-status.
EMB: dict[str, np.ndarray] = {
    "king":      np.array([0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
    "queen":     np.array([0.9, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.8]),
    "prince":    np.array([0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]),
    "princess":  np.array([0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.25]),
    "man":       np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "woman":     np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "uncle":     np.array([0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
    "aunt":      np.array([0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2]),
    "cat":       np.array([0.0, 0.0, 0.0, 0.9, 0.8, 0.0, 0.0, 0.0]),
    "dog":       np.array([0.0, 0.0, 0.0, 0.9, 0.7, 0.1, 0.0, 0.0]),
    "kitten":    np.array([0.0, 0.0, 0.0, 0.95, 0.85, 0.0, 0.0, 0.0]),
    "nurse":     np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "doctor":    np.array([0.0, 0.3, 0.25, 0.0, 0.0, 0.9, 0.1, 0.1]),
    "engineer":  np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.1, 0.9, 0.1]),
    "teacher":   np.array([0.0, 0.1, 0.3, 0.0, 0.0, 0.5, 0.2, 0.0]),
}


def vec(word: str) -> np.ndarray:
    return EMB[word]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def nearest(word: str, table: dict, k: int = 3) -> list[tuple[str, float]]:
    target = table[word]
    scored = [
        (w, cosine(target, v))
        for w, v in table.items()
        if w != word
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]


def sort_best_first(pair: tuple) -> tuple:
    """Sort key for (word, score) pairs: highest score first, then A to Z.

    Python sorts tuples left to right and always ascending, so negating the
    score turns "highest first" into "smallest first" without a second pass.
    """
    word, score = pair
    return (-score, word)


def analogy(a: str, b: str, c: str, table: dict, k: int = 1) -> list[tuple[str, float]]:
    vec_a = table[a]
    vec_b = table[b]
    vec_c = table[c]

    # The step from a to b, applied starting at c.
    target = vec_b - vec_a + vec_c

    scored = []
    for word in table:
        if word == a or word == b or word == c:
            continue
        score = cosine(target, table[word])
        scored.append((word, score))

    # Best score first; when two tie, the alphabetically earlier word wins.
    scored.sort(key=sort_best_first)
    return scored[:k]


def bias_score(word: str, pos: str, neg: str, table: dict) -> float:
    direction = table[pos] - table[neg]
    return cosine(table[word], direction)


def load_pretrained():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        print("sentence-transformers not available, skipping pretrained demo.")
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # noqa: BLE001
        print(f"Could not load pretrained model (offline?): {e}")
        return None


def _demo() -> None:
    print("=" * 60)
    print("Word Embeddings, neighbors, analogies & bias probing")
    print("=" * 60)

    print("\nNearest neighbors of 'king':")
    for w, s in nearest("king", EMB, k=3):
        print(f"   {s:5.3f}  {w}")

    print("\nAnalogy  man : king :: woman : ?")
    for w, s in analogy("man", "king", "woman", EMB, k=1):
        print(f"   {s:5.3f}  {w}")

    print("\nBias probe along the (woman - man) direction:")
    for occ in ["nurse", "doctor", "engineer", "teacher"]:
        s = bias_score(occ, "woman", "man", EMB)
        lean = "-> woman" if s > 0 else "-> man"
        print(f"   {occ:9s} {s:+.3f}  {lean}")
    print("\n(Illustrative toy vectors; real embeddings show the same patterns.)")


if __name__ == "__main__":
    _demo()
