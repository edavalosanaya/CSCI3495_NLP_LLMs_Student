"""W5C2 starter, build scaled dot-product & multi-head attention from scratch.

Work through the lab in `README.md`. Each STEP has its own check:
    python -m pytest weeks/week-05/class-02/exercise/test_attention_lab.py -k step1 -q

RUN the file at any point to see how far you have got (it prints the last
milestone that works and which one is next):
  docker compose -f docker/docker-compose.yml run --rm --no-deps \
      course python weeks/week-05/class-02/exercise/attention_lab.py
Then verify with the tests:
  docker compose -f docker/docker-compose.yml run --rm --no-deps \
      course python -m pytest weeks/week-05/class-02/exercise/test_attention_lab.py -q
"""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #   numerically-stable softmax along `axis` (subtract the max first)
    raise NotImplementedError


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q:(...,Tq,d) K:(...,Tk,d) V:(...,Tk,dv). Return (output, weights).
    scores = Q K^T / sqrt(d_k); apply mask (keep where True); softmax; weight V."""
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #   Use np.swapaxes(K, -1, -2), NOT K.T: step 5 passes 3-D arrays.
    raise NotImplementedError


def causal_mask(T: int) -> np.ndarray:
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #   lower-triangular boolean mask of shape (T, T)
    raise NotImplementedError


def split_heads(X, num_heads):
    # TODO (STEP 4): implement. Check with: pytest -k step4
    #   (T, d_model) -> (num_heads, T, d_head); reshape THEN transpose
    raise NotImplementedError


def combine_heads(X):
    # TODO (STEP 4): implement. Check with: pytest -k step4
    #   (num_heads, T, d_head) -> (T, num_heads*d_head); exact inverse of split
    raise NotImplementedError


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    # TODO (STEP 5): implement. Check with: pytest -k step5
    #   project to Q,K,V; split into heads; attend ONCE on the 3-D arrays;
    #   combine; project with Wo
    raise NotImplementedError


def main() -> int:
    """Progressive milestone runner: shows how far your implementation gets."""
    rng = np.random.default_rng(0)
    T, d = 4, 8
    X = rng.normal(size=(T, d))

    try:
        Q = K = V = X
        out, w = scaled_dot_product_attention(Q, K, V)
        print("MILESTONE 1  softmax + scaled dot-product attention: WORKS")
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
