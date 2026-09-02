"""Tests for W5C2 attention_lab. Set ATTN_FROM=solution to test the reference.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-05/class-02/exercise/test_attention_lab.py -k step1 -q
"""
import importlib.util
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (_HERE.parent / "solutions" / "attention_lab.py"
        if os.environ.get("ATTN_FROM") == "solution"
        else _HERE / "attention_lab.py")
_spec = importlib.util.spec_from_file_location("attn_uut", _SRC)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_given_softmax_sums_to_one():
    w = m.softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(w.sum(), 1.0) and np.all(w > 0)


def test_step1_uniform_scores_average_values():
    # Equal Q·K for all keys -> uniform weights -> output is mean of V
    Q = np.zeros((1, 4)); K = np.zeros((3, 4))
    V = np.array([[1.0, 0], [3.0, 0], [5.0, 0]])
    out, w = m.scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w, 1 / 3)
    assert np.allclose(out, [3.0, 0])  # mean


def test_step2_causal_mask_zeros_future():
    T, d = 4, 8
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(T, d)); K = rng.normal(size=(T, d)); V = rng.normal(size=(T, d))
    _, w = m.scaled_dot_product_attention(Q, K, V, mask=m.causal_mask(T))
    # upper triangle (future) must be ~0
    assert np.allclose(np.triu(w, k=1), 0.0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_given_split_combine_roundtrip():
    X = np.arange(2 * 8).reshape(2, 8).astype(float)
    assert np.allclose(m.combine_heads(m.split_heads(X, 4)), X)


def test_step3_multihead_shape_and_single_head_matches_plain():
    rng = np.random.default_rng(1)
    T, d = 5, 12
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = (np.eye(d) for _ in range(4))
    out = m.multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=3)
    assert out.shape == (T, d)
    # With identity projections and 1 head, MHA == plain attention on X
    out1 = m.multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=1)
    plain, _ = m.scaled_dot_product_attention(X, X, X)
    assert np.allclose(out1, plain)
