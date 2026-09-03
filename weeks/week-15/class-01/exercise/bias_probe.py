#!/usr/bin/env python3
"""W15C1, Bias probing in word embeddings (STARTER)."""
from __future__ import annotations

import numpy as np

from embeddings import EMBEDDINGS, dim  # provided toy vectors


def get_vector(word: str) -> np.ndarray:
    """Return the embedding for `word`, or raise KeyError if unknown."""
    return np.asarray(EMBEDDINGS[word], dtype=float)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: any vector.
        b: any vector of the same length.

    Returns:
        A float in [-1, 1]. 0.0 when either vector has zero length, since a
        vector with no direction has no angle to measure.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The formula is in README section 3.
    #
    #   measure each vector's length, and if either is zero there is no angle
    #   otherwise divide their dot product by the two lengths
    #
    raise NotImplementedError


def association(word: str, set_a: list[str], set_b: list[str]) -> float:
    """Which of two attribute sets a single word sits closer to.

    Args:
        word: the word being probed, e.g. "engineer".
        set_a: one pole of the attribute axis, e.g. male-coded words.
        set_b: the other pole, e.g. female-coded words. The two sets need not
            be the same size, which is why each side is averaged.

    Returns:
        A float. Positive means the word is on average closer to set A,
        negative closer to set B, and near zero means the axis says nothing
        about this word.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The formula is in README section 4.
    #
    #   look up the word's own vector
    #   average its cosine against every word in the first set
    #   do the same for the second set
    #   the answer is the difference between those two averages
    #
    raise NotImplementedError


def effect(targets_x: list[str], targets_y: list[str],
           attr_a: list[str], attr_b: list[str]) -> float:
    """A WEAT-style summary over two groups of target words.

    Args:
        targets_x: the first group of words, e.g. career-coded.
        targets_y: the second group, e.g. family-coded.
        attr_a: one pole of the attribute axis.
        attr_b: the other pole.

    Returns:
        A float. Large and positive means X leans toward A while Y leans toward
        B, which is the stereotype-aligned pattern. Swapping attr_a and attr_b
        flips the sign; swapping the targets flips it too.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   The formula is in README section 5, and it is step 2 applied twice.
    #
    #   average the association of every word in the first target group
    #   do the same for the second group
    #   the answer is the difference between those averages
    #
    raise NotImplementedError


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
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        # A classic probe shape: career/family vs. male/female terms.
        X = ["engineer", "programmer", "scientist"]   # "career"-coded targets
        Y = ["nurse", "teacher", "homemaker"]         # "family"/care-coded targets
        A = ["man", "he", "male"]
        B = ["woman", "she", "female"]
        print(report(X, Y, A, B, labels=("career", "care", "male", "female")))
        print("Reflect: the toy vectors were *constructed* to show this pattern.")
        print("Real embeddings (Week 3) show similar, well-documented biases.")
    except NotImplementedError:
        print("bias_probe.py is not finished yet: fill in the next TODO in this file, then re-run.")
