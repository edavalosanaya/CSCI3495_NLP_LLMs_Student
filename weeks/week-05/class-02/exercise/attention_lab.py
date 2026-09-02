"""W5C2 starter, build scaled dot-product & multi-head attention from scratch."""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """The Transformer's attention: match queries to keys, blend values.

    Args:
        Q: shape (..., Tq, d_k). One query row per output position.
        K: shape (..., Tk, d_k). One key row per input position.
        V: shape (..., Tk, d_v). What gets blended. Tk must match K's.
        mask: optional boolean, broadcastable to (..., Tq, Tk). True means the
            position MAY be looked at; False means it must be hidden.

    Returns:
        (output, weights). output is (..., Tq, d_v). weights is
        (..., Tq, Tk) and each row sums to 1, so it can be inspected.
        Leading axes are batch/head axes and pass straight through.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The formula is in README section 2.
    #
    #   multiply the queries by the keys transposed on their LAST TWO axes,
    #       so any batch or head axes in front are left alone
    #   divide by the square root of the key dimension
    #   where the mask says a position is hidden, replace the score with a
    #       very large negative number so the softmax sends it to zero
    #   normalize each row of scores into a distribution over the keys
    #   blend the values by those weights, and return them alongside
    #
    #   Scale BEFORE the softmax, not after: the whole point is to stop large
    #   dot products from saturating it. `softmax` is given above.
    #
    raise NotImplementedError


def causal_mask(T: int) -> np.ndarray:
    """Build the mask that stops a position from reading the future.

    Args:
        T: the sequence length.

    Returns:
        A (T, T) boolean array. True means "may look here". Row t may look at
        columns 0 through t inclusive, so a position always sees itself.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   One line. You want a square array of booleans where everything on and
    #   below the diagonal is True and everything above it is False. NumPy has
    #   a function that keeps the lower triangle of an array.
    #
    raise NotImplementedError


def split_heads(X, num_heads):
    """GIVEN. (T, d_model) -> (num_heads, T, d_head)."""
    T, d_model = X.shape
    d_head = d_model // num_heads
    return X.reshape(T, num_heads, d_head).transpose(1, 0, 2)


def combine_heads(X):
    """GIVEN. (num_heads, T, d_head) -> (T, d_model), the inverse of split_heads."""
    num_heads, T, d_head = X.shape
    return X.transpose(1, 0, 2).reshape(T, num_heads * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """Run several attention heads over the same input and merge them.

    Args:
        X: shape (T, d_model). The sequence, one row per position.
        Wq, Wk, Wv: each (d_model, d_model). Project X into queries, keys and
            values. All three see the same X: that is what makes it SELF
            attention.
        Wo: (d_model, d_model). Mixes the concatenated heads back together.
        num_heads: how many heads to split d_model across. Must divide it.
        mask: optional, passed straight through to the attention.

    Returns:
        Shape (T, d_model), the same shape as X, which is what lets these
        blocks stack.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   Every piece already exists; this step assembles them.
    #
    #   project X three times, into queries, keys and values
    #   split each projection into heads
    #   attend, passing the mask through, so every head attends at once
    #   put the heads back side by side, then mix them with the output matrix
    #
    raise NotImplementedError


def main() -> int:
    """Progressive milestone runner: shows how far your implementation gets."""
    rng = np.random.default_rng(0)
    T, d = 4, 8
    X = rng.normal(size=(T, d))

    try:
        Q = K = V = X
        out, w = scaled_dot_product_attention(Q, K, V)
        print("MILESTONE 1  scaled dot-product attention: WORKS")
        print("  weights row 0:", np.round(w[0], 3), " (sums to", round(float(w[0].sum()), 3), ")")
    except NotImplementedError:
        print("MILESTONE 1  scaled_dot_product_attention: not written yet (Step 2)")
        return 0

    try:
        m = causal_mask(T)
        _, wm = scaled_dot_product_attention(Q, K, V, mask=m)
        print("MILESTONE 2  causal mask: WORKS")
        print("  masked weights (rounded); note the zeros ABOVE the diagonal, the future:")
        for row in np.round(wm, 2):
            print("   ", row)
    except NotImplementedError:
        print("MILESTONE 2  causal_mask: not written yet (Step 3), keep going")
        return 0

    try:
        Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.1 for _ in range(4))
        out = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=2, mask=causal_mask(T))
        print("MILESTONE 3  multi-head attention: WORKS, output shape", out.shape)
        print("\nAll milestones run. Now make the tests pass.")
    except NotImplementedError:
        print("MILESTONE 3  multi_head_attention: not written yet (Step 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
