#!/usr/bin/env python3
"""W3C2 starter, explore word embeddings: neighbors, analogies, bias."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# A tiny toy embedding space (8-D), arranged so that:
#  - semantic neighbors are close (king/queen/prince; man/woman),
#  - a consistent "gender" offset exists (man->woman ~ king->queen),
#  - an illustrative occupation-gender association is present (for the bias probe).
# These are illustrative, NOT trained, they let you test your vector math.
# ---------------------------------------------------------------------------
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
    """Return the embedding for `word` (raises KeyError if absent)."""
    return EMB[word]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """GIVEN. Cosine similarity, 0.0 if either vector has zero length."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def nearest(word: str, table: dict, k: int = 3) -> list[tuple[str, float]]:
    """GIVEN. The k most similar words to `word`, itself excluded, best first."""
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
    """Solve "a is to b as c is to ?" by arithmetic on the word vectors.

    Args:
        a: the first word of the source pair (classic example: "man").
        b: the second word of the source pair ("king"). The step from a to b
            is the relationship being transferred.
        c: the word the same step is applied to ("woman").
        table: word -> vector, each vector a numpy array. Every candidate
            answer is drawn from its keys.
        k: how many candidates to return.

    Returns:
        Up to k (word, similarity) pairs, best first, ties broken by the word
        itself. a, b and c are never returned: they sit closest to the target
        vector by construction, and returning them answers nothing.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The target vector is in README section 3.
    #
    #   build the target by stepping from a to b, then applying that step at c
    #   score every word in the table against the target with cosine
    #   leave out the three input words
    #   sort with the given sort_best_first: best score first, ties A to Z
    #   hand back the first k pairs
    #
    raise NotImplementedError


def bias_score(word: str, pos: str, neg: str, table: dict) -> float:
    """How far a word leans along the axis running from `neg` to `pos`.

    Args:
        word: the word being probed, e.g. "nurse".
        pos: the word defining the positive end of the axis, e.g. "she".
        neg: the word defining the negative end, e.g. "he".
        table: word -> vector, unused here beyond looking the words up.

    Returns:
        A float in [-1, 1]. Positive leans toward `pos`, negative toward `neg`,
        and near 0 means the axis says nothing about this word. Swapping pos
        and neg flips the sign.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The formula is in README section 4.
    #
    #   the axis is the vector that points from neg to pos
    #   the answer is how closely the word's own vector aligns with that axis
    #
    raise NotImplementedError


def load_pretrained():
    """OPTIONAL: load real pretrained vectors if available; else return None.

    This is for the stretch goal. It tries sentence-transformers (which may need
    a one-time model download). If anything is missing or offline, it returns
    None with a clear message instead of crashing.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        print("sentence-transformers not available, skipping pretrained demo.")
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
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
    try:
        _demo()
    except NotImplementedError:
        print("embeddings.py is not finished yet: fill in the next TODO in this file, then re-run.")
