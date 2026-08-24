"""Tests for HW3, Attention & the Transformer (from scratch).

Runs against the student's starter file by default. To check the reference
solution (used by the course test sweep):

    HW3_FROM=solution  python -m pytest homeworks/hw3 -q

The suite gracefully SKIPS while the starter still raises NotImplementedError.
All dimensions are tiny so the whole suite runs in well under a second.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE / "solutions" / "transformer.py"
    if os.environ.get("HW3_FROM") == "solution"
    else _HERE / "transformer.py"
)
_spec = importlib.util.spec_from_file_location("hw3_under_test", _SRC)
T = importlib.util.module_from_spec(_spec)
sys.modules["hw3_under_test"] = T
_spec.loader.exec_module(T)


def _implemented() -> bool:
    try:
        T.softmax(np.array([0.0, 0.0]))
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="HW3 not implemented yet (fill in the TODOs)"
)


# --------------------------------------------------------------------------
# Task 1, softmax
# --------------------------------------------------------------------------
def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    s = T.softmax(x, axis=-1)
    assert np.allclose(s.sum(axis=-1), 1.0)


def test_softmax_uniform_on_equal_logits():
    s = T.softmax(np.zeros(4))
    assert np.allclose(s, 0.25)


def test_softmax_numerically_stable():
    # Large logits must not overflow.
    s = T.softmax(np.array([1000.0, 1000.0, 1000.0]))
    assert np.allclose(s, 1 / 3)
    assert np.all(np.isfinite(s))


# --------------------------------------------------------------------------
# Task 2, scaled dot-product attention
# --------------------------------------------------------------------------
def test_attention_weights_sum_to_one():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((3, 4))
    K = rng.standard_normal((5, 4))
    V = rng.standard_normal((5, 2))
    out, w = T.scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 2)
    assert w.shape == (3, 5)
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert np.all(w >= 0.0)


def test_attention_known_value_uniform():
    # Identical keys -> equal scores -> uniform weights -> output is mean of V.
    Q = np.array([[1.0, 0.0]])
    K = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    V = np.array([[1.0], [2.0], [3.0]])
    out, w = T.scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w, 1 / 3)
    assert np.allclose(out, [[2.0]])  # mean of 1,2,3


def test_causal_mask_blocks_future():
    m = T.causal_mask(3)
    assert m.shape == (3, 3)
    # diagonal and below are 0
    assert m[0, 0] == 0.0 and m[2, 1] == 0.0 and m[2, 0] == 0.0
    # strictly upper triangle is very negative
    assert m[0, 1] < -1e8 and m[0, 2] < -1e8 and m[1, 2] < -1e8


def test_causal_attention_ignores_future():
    rng = np.random.default_rng(1)
    n = 4
    Q = rng.standard_normal((n, 3))
    K = rng.standard_normal((n, 3))
    V = rng.standard_normal((n, 3))
    _, w = T.scaled_dot_product_attention(Q, K, V, T.causal_mask(n))
    # Position i must put ~0 weight on any future position j > i.
    for i in range(n):
        for j in range(i + 1, n):
            assert w[i, j] < 1e-6
        assert math.isclose(w[i, : i + 1].sum(), 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------
# Task 3, multi-head attention
# --------------------------------------------------------------------------
def _make_mha(d_model=4, num_heads=2, seed=0):
    rng = np.random.default_rng(seed)
    Wq = rng.standard_normal((d_model, d_model))
    Wk = rng.standard_normal((d_model, d_model))
    Wv = rng.standard_normal((d_model, d_model))
    Wo = rng.standard_normal((d_model, d_model))
    return T.MultiHeadAttention(d_model, num_heads, Wq, Wk, Wv, Wo)


def test_split_combine_roundtrip():
    mha = _make_mha()
    x = np.arange(3 * 4, dtype=np.float64).reshape(3, 4)
    heads = mha._split_heads(x)
    assert heads.shape == (2, 3, 2)  # (num_heads, seq, d_head)
    back = mha._combine_heads(heads)
    assert np.allclose(back, x)


def test_mha_output_shape_and_weights():
    mha = _make_mha(d_model=4, num_heads=2)
    x = np.random.default_rng(2).standard_normal((5, 4))
    out, w = mha.forward(x)
    assert out.shape == (5, 4)
    assert w.shape == (2, 5, 5)  # (num_heads, seq, seq)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_mha_causal_mask_applies_per_head():
    mha = _make_mha(d_model=4, num_heads=2)
    n = 4
    x = np.random.default_rng(3).standard_normal((n, 4))
    _, w = mha.forward(x, T.causal_mask(n))
    # every head respects causality
    for h in range(2):
        for i in range(n):
            for j in range(i + 1, n):
                assert w[h, i, j] < 1e-6


def test_mha_single_head_matches_plain_attention():
    # With one head, MHA = (xWq, xWk, xWv) -> attention -> @ Wo.
    d = 3
    rng = np.random.default_rng(7)
    Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) for _ in range(4))
    mha = T.MultiHeadAttention(d, 1, Wq, Wk, Wv, Wo)
    x = rng.standard_normal((4, d))
    out, _ = mha.forward(x)
    ref_out, _ = T.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv)
    assert np.allclose(out, ref_out @ Wo)


# --------------------------------------------------------------------------
# Task 4, positional encoding
# --------------------------------------------------------------------------
def test_positional_encoding_shape_and_range():
    pe = T.positional_encoding(10, 8)
    assert pe.shape == (10, 8)
    assert np.all(pe <= 1.0 + 1e-9) and np.all(pe >= -1.0 - 1e-9)


def test_positional_encoding_known_values():
    pe = T.positional_encoding(3, 4)
    # pos=0: sin(0)=0, cos(0)=1 across all channels
    assert np.allclose(pe[0, 0::2], 0.0)
    assert np.allclose(pe[0, 1::2], 1.0)
    # pos=1, channel 0: sin(1 / 10000^0) = sin(1)
    assert math.isclose(pe[1, 0], math.sin(1.0), rel_tol=1e-9)
    assert math.isclose(pe[1, 1], math.cos(1.0), rel_tol=1e-9)


def test_positional_encoding_distinct_positions():
    pe = T.positional_encoding(5, 8)
    # Different positions get different encodings.
    assert not np.allclose(pe[0], pe[1])


# --------------------------------------------------------------------------
# Task 5, layer norm, FFN, encoder block
# --------------------------------------------------------------------------
def test_layer_norm_zero_mean_unit_var():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    g = np.ones(4)
    b = np.zeros(4)
    y = T.layer_norm(x, g, b)
    assert math.isclose(float(y.mean()), 0.0, abs_tol=1e-6)
    # variance ~ 1 (with eps it's slightly under)
    assert abs(float(y.var()) - 1.0) < 1e-3


def test_layer_norm_affine():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    g = np.full(4, 2.0)
    b = np.full(4, 5.0)
    y = T.layer_norm(x, g, b)
    # shifted by beta -> mean ~ 5
    assert math.isclose(float(y.mean()), 5.0, abs_tol=1e-5)


def test_relu():
    assert np.allclose(T.relu(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 2.0])


def test_feedforward_shape():
    d, hidden = 4, 6
    rng = np.random.default_rng(0)
    ffn = T.FeedForward(
        rng.standard_normal((d, hidden)), rng.standard_normal(hidden),
        rng.standard_normal((hidden, d)), rng.standard_normal(d),
    )
    out = ffn.forward(rng.standard_normal((3, d)))
    assert out.shape == (3, d)


def test_encoder_block_shape_preserved():
    d, heads, hidden, seq = 4, 2, 8, 5
    rng = np.random.default_rng(123)
    mha = T.MultiHeadAttention(
        d, heads, rng.standard_normal((d, d)), rng.standard_normal((d, d)),
        rng.standard_normal((d, d)), rng.standard_normal((d, d)),
    )
    ffn = T.FeedForward(
        rng.standard_normal((d, hidden)), rng.standard_normal(hidden),
        rng.standard_normal((hidden, d)), rng.standard_normal(d),
    )
    block = T.EncoderBlock(
        mha, ffn, (np.ones(d), np.zeros(d)), (np.ones(d), np.zeros(d))
    )
    x = rng.standard_normal((seq, d))
    out = block.forward(x)
    assert out.shape == (seq, d)
    assert np.all(np.isfinite(out))


def test_encoder_block_layernorm_output():
    # Post-LN block: the final output rows should be ~zero-mean (gamma=1,beta=0).
    d, heads, hidden, seq = 4, 2, 8, 4
    rng = np.random.default_rng(9)
    mha = T.MultiHeadAttention(
        d, heads, rng.standard_normal((d, d)), rng.standard_normal((d, d)),
        rng.standard_normal((d, d)), rng.standard_normal((d, d)),
    )
    ffn = T.FeedForward(
        rng.standard_normal((d, hidden)), rng.standard_normal(hidden),
        rng.standard_normal((hidden, d)), rng.standard_normal(d),
    )
    block = T.EncoderBlock(
        mha, ffn, (np.ones(d), np.zeros(d)), (np.ones(d), np.zeros(d))
    )
    out = block.forward(rng.standard_normal((seq, d)))
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
