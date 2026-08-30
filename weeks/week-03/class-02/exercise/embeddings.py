#!/usr/bin/env python3
"""W3C2 starter, explore word embeddings: neighbors, analogies, bias.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-03/class-02/exercise/test_embeddings.py -k step1 -q

When all four steps are done, the demo runs:
    python weeks/week-03/class-02/exercise/embeddings.py

We ship a small, hand-built embedding table (EMB) so everything runs offline,
CPU-only, and deterministically. The math you write here is exactly what you'd
run on real word2vec/GloVe vectors. The optional `load_pretrained()` helper at
the bottom shows how to use real vectors if they're available, and degrades
gracefully if they aren't.
"""
from __future__ import annotations
import math

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
EMB: dict[str, list[float]] = {
    "king":      [0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
    "queen":     [0.9, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.8],
    "prince":    [0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
    "princess":  [0.8, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.25],
    "man":       [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    "woman":     [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.1],
    "uncle":     [0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "aunt":      [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2],
    "cat":       [0.0, 0.0, 0.0, 0.9, 0.8, 0.0, 0.0, 0.0],
    "dog":       [0.0, 0.0, 0.0, 0.9, 0.7, 0.1, 0.0, 0.0],
    "kitten":    [0.0, 0.0, 0.0, 0.95, 0.85, 0.0, 0.0, 0.0],
    "nurse":     [0.0, 0.0, 0.5, 0.0, 0.0, 0.9, 0.0, 0.0],
    "doctor":    [0.0, 0.3, 0.25, 0.0, 0.0, 0.9, 0.1, 0.1],
    "engineer":  [0.0, 0.5, 0.0, 0.0, 0.0, 0.1, 0.9, 0.1],
    "teacher":   [0.0, 0.1, 0.3, 0.0, 0.0, 0.5, 0.2, 0.0],
}


def vec(word: str) -> np.ndarray:
    """Return the embedding for `word` as a numpy array (raises KeyError if absent)."""
    return np.asarray(EMB[word], dtype=float)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors. Return 0.0 if either norm is 0."""
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def nearest(word: str, table: dict, k: int = 3) -> list[tuple[str, float]]:
    """Return the k most cosine-similar words to `word` (excluding `word` itself).

    Each result is (other_word, similarity), sorted by similarity descending.
    """
    # GIVEN (STEP 2): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    target = np.asarray(table[word], dtype=float)
    scored = [
        (w, cosine(target, np.asarray(v, dtype=float)))
        for w, v in table.items()
        if w != word
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]


def analogy(a: str, b: str, c: str, table: dict, k: int = 1) -> list[tuple[str, float]]:
    """Solve 'a is to b as c is to ?' via vec(b) - vec(a) + vec(c).

    Return the top-k closest words by cosine to that target vector, EXCLUDING
    the three input words a, b, c. (Classic example: a=man, b=king, c=woman.)
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    # Without the exclusion this returns "king", not "queen".
    raise NotImplementedError


def bias_score(word: str, pos: str, neg: str, table: dict) -> float:
    """Projection of `word` onto the (pos - neg) direction, via cosine.

    cosine(vec(word), vec(pos) - vec(neg)). A positive score leans toward `pos`,
    negative toward `neg`. Use it to probe, e.g., how 'nurse' vs 'engineer'
    align with the woman - man direction.
    """
    # TODO (STEP 4): implement. Check with: pytest -k step4
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
        print("embeddings.py is not implemented yet, fill in the TODOs, then re-run.")
