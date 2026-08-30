"""W5C2 reference solution, scaled dot-product & multi-head attention (NumPy)."""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q:(...,Tq,d) K:(...,Tk,d) V:(...,Tk,dv). Returns (output, weights)."""
    d_k = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)  # (...,Tq,Tk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_mask(T: int) -> np.ndarray:
    """Lower-triangular True mask: position t may attend to <= t."""
    return np.tril(np.ones((T, T), dtype=bool))


def split_heads(X, num_heads):
    """(T, d_model) -> (num_heads, T, d_head)."""
    T, d_model = X.shape
    d_head = d_model // num_heads
    return X.reshape(T, num_heads, d_head).transpose(1, 0, 2)


def combine_heads(X):
    """(num_heads, T, d_head) -> (T, num_heads*d_head)."""
    num_heads, T, d_head = X.shape
    return X.transpose(1, 0, 2).reshape(T, num_heads * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """X:(T,d_model); W*:(d_model,d_model). Returns (T, d_model)."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    Qh, Kh, Vh = (split_heads(M, num_heads) for M in (Q, K, V))
    out_h, _ = scaled_dot_product_attention(Qh, Kh, Vh, mask=mask)
    return combine_heads(out_h) @ Wo


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
        print("MILESTONE 1  softmax / scaled_dot_product_attention: not implemented yet")
        return 0

    try:
        m = causal_mask(T)
        _, wm = scaled_dot_product_attention(Q, K, V, mask=m)
        print("MILESTONE 2  causal mask: WORKS")
        print("  masked weights (rounded); note the zeros ABOVE the diagonal, the future:")
        for row in np.round(wm, 2):
            print("   ", row)
    except NotImplementedError:
        print("MILESTONE 2  causal_mask: not implemented yet, keep going")
        return 0

    try:
        Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.1 for _ in range(4))
        out = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=2, mask=causal_mask(T))
        print("MILESTONE 3  multi-head attention: WORKS, output shape", out.shape)
        print("\nAll milestones run. Now make the tests pass.")
    except NotImplementedError:
        print("MILESTONE 3  split/combine heads / multi_head_attention: not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
