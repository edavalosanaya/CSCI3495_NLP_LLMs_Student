#!/usr/bin/env python3
"""W15C1, Provided toy word embeddings (offline, deterministic).

These are SMALL, HAND-CONSTRUCTED 8-d vectors, not trained on real text. They
are built so that a bias probe will *find* a stereotype-aligned pattern, letting
students measure it transparently without any downloads.

How they're built (so it's not magic):
  * dim 0 = a "gender" axis: +x = male-coded, -x = female-coded.
  * dim 1 = a "career vs. care" axis: +y = career-coded, -y = care-coded.
  * remaining dims = small deterministic noise so vectors aren't degenerate.

We then *correlate* the two axes for the target words (career words also placed
on the male side, care words on the female side), mirroring the real, documented
association in trained embeddings (e.g., word2vec/GloVe, Week 3).

In Week 3 you probed REAL embeddings; here we isolate the mechanism on toy data.
"""
from __future__ import annotations

import numpy as np

_DIM = 8
_rng = np.random.default_rng(20260610)  # fixed seed -> fully deterministic


def _vec(gender: float, career: float) -> list[float]:
    """Build an 8-d vector from a gender coord and a career coord + tiny noise."""
    base = np.zeros(_DIM)
    base[0] = gender          # gender axis
    base[1] = career          # career-vs-care axis
    base[2:] = _rng.normal(0.0, 0.05, size=_DIM - 2)  # small deterministic noise
    return base.tolist()


# gender: +1 male, -1 female, 0 neutral.   career: +1 career, -1 care, 0 neutral.
EMBEDDINGS: dict[str, list[float]] = {
    # --- attribute words (define the gender axis) ---
    "man":    _vec(+1.0, 0.0),
    "he":     _vec(+1.0, 0.0),
    "male":   _vec(+1.0, 0.0),
    "woman":  _vec(-1.0, 0.0),
    "she":    _vec(-1.0, 0.0),
    "female": _vec(-1.0, 0.0),

    # --- target words: career-coded, leaning male (the stereotype to expose) ---
    "engineer":   _vec(+0.6, +1.0),
    "programmer": _vec(+0.6, +1.0),
    "scientist":  _vec(+0.5, +1.0),

    # --- target words: care-coded, leaning female ---
    "nurse":     _vec(-0.6, -1.0),
    "teacher":   _vec(-0.4, -0.8),
    "homemaker": _vec(-0.7, -1.0),

    # --- a few neutral words for contrast / experiments ---
    "tree":   _vec(0.0, 0.0),
    "river":  _vec(0.0, 0.0),
    "music":  _vec(0.0, 0.0),
}


def dim() -> int:
    """Embedding dimensionality."""
    return _DIM
