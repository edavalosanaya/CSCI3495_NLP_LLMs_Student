"""HW3 starter, Attention & the Transformer, from scratch in NumPy.

You will implement the core building blocks of the Transformer architecture
(Vaswani et al., 2017, "Attention Is All You Need", arXiv:1706.03762), using
ONLY NumPy. Everything is tiny so the tests run in seconds on a CPU.

Run the tests:
    docker compose -f docker/docker-compose.yml run --rm course \
        python -m pytest homeworks/hw3 -q

Conventions
-----------
- Sequences are arrays of shape (seq_len, d_model) unless a leading batch
  dimension is noted.
- Use float64 internally for numerical stability in the tests.
- Do NOT use torch.nn or any framework attention; implement the math yourself.
"""
# Each TODO below names its README step. Check one step with:
#     python -m pytest homeworks/hw3 -q -k step3      (or step1, step2, ...)
# and the whole assignment with:
#     python -m pytest homeworks/hw3 -q

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Step 1, Softmax
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along ``axis``.

    Subtract the per-slice max before exponentiating so large logits don't
    overflow. Output sums to 1 along ``axis``.
    """
    # TODO (STEP 1): implement (subtract max, exp, normalize)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2, Scaled dot-product attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention.

        Attention(Q, K, V) = softmax( (Q Kᵀ) / sqrt(d_k) + mask ) V

    Shapes:
        Q: (..., n_q, d_k)
        K: (..., n_k, d_k)
        V: (..., n_k, d_v)
        mask (optional): broadcastable to (..., n_q, n_k); positions to forbid
                         should hold a large negative value (e.g. -1e9), allowed
                         positions hold 0.

    Returns:
        output: (..., n_q, d_v)
        weights: (..., n_q, n_k), the attention probabilities (rows sum to 1).
    """
    # TODO (STEP 2): compute scores = Q @ K^T / sqrt(d_k); add mask if given;
    #       weights = softmax(scores, axis=-1); output = weights @ V
    raise NotImplementedError


def causal_mask(seq_len: int) -> np.ndarray:
    """Return a (seq_len, seq_len) additive mask for autoregressive attention.

    Entry (i, j) is 0.0 if j <= i (token i may attend to token j) and a large
    negative number (-1e9) if j > i (no attending to the future).
    """
    # TODO (STEP 3): build with np.triu / np.tril
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 3, Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention:
    """Multi-head self-attention with explicit weight matrices.

    Given d_model and num_heads (d_model must be divisible by num_heads), this
    projects inputs into per-head queries/keys/values, runs scaled dot-product
    attention per head, concatenates, and applies an output projection.

    Weight matrices (all shape (d_model, d_model)) are provided at construction
    so the forward pass is deterministic and testable:
        W_q, W_k, W_v, W_o
    """

    def __init__(self, d_model: int, num_heads: int, W_q, W_k, W_v, W_o):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = np.asarray(W_q, dtype=np.float64)
        self.W_k = np.asarray(W_k, dtype=np.float64)
        self.W_v = np.asarray(W_v, dtype=np.float64)
        self.W_o = np.asarray(W_o, dtype=np.float64)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(seq, d_model) -> (num_heads, seq, d_head)."""
        # TODO (STEP 4): reshape to (seq, num_heads, d_head) then transpose to
        #       (num_heads, seq, d_head)
        raise NotImplementedError

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(num_heads, seq, d_head) -> (seq, d_model). Inverse of _split_heads."""
        # TODO (STEP 4): transpose to (seq, num_heads, d_head) then reshape to (seq, d_model)
        raise NotImplementedError

    def forward(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run multi-head self-attention on a single sequence x: (seq, d_model).

        Steps:
          1. Q = x @ W_q, K = x @ W_k, V = x @ W_v   (each (seq, d_model)).
          2. Split each into heads: (num_heads, seq, d_head).
          3. Per-head scaled dot-product attention (mask broadcasts over heads).
          4. Combine heads back to (seq, d_model).
          5. Output projection: out = combined @ W_o.

        Returns (output: (seq, d_model), weights: (num_heads, seq, seq)).
        """
        # TODO (STEP 5): implement
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 4, Positional encoding
# ---------------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encodings (Vaswani et al., 2017, Eq. 3.5).

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Returns an array of shape (seq_len, d_model). Assume d_model is even.
    """
    # TODO (STEP 6): implement (build pos column, div_term, fill even/odd channels)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 5, Building blocks: LayerNorm, FFN, and an encoder block
# ---------------------------------------------------------------------------
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization over the last axis.

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    mean and var are computed over the last axis (the feature dimension).
    gamma and beta have shape (d_model,).
    """
    # TODO (STEP 7): implement
    raise NotImplementedError


def relu(x: np.ndarray) -> np.ndarray:
    """Elementwise ReLU."""
    # TODO (STEP 7): implement
    raise NotImplementedError


class FeedForward:
    """Position-wise feed-forward network: ReLU(x W1 + b1) W2 + b2."""

    def __init__(self, W1, b1, W2, b2):
        self.W1 = np.asarray(W1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.W2 = np.asarray(W2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO (STEP 7): implement relu(x @ W1 + b1) @ W2 + b2
        raise NotImplementedError


class EncoderBlock:
    """A single Transformer encoder block (post-LN, as in the original paper).

        a = LayerNorm(x + MultiHeadAttention(x))
        out = LayerNorm(a + FeedForward(a))
    """

    def __init__(self, mha: MultiHeadAttention, ffn: FeedForward, ln1_params, ln2_params):
        self.mha = mha
        self.ffn = ffn
        # each *_params is a tuple (gamma, beta)
        self.ln1_gamma, self.ln1_beta = ln1_params
        self.ln2_gamma, self.ln2_beta = ln2_params

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Run the encoder block on x: (seq, d_model). Returns (seq, d_model)."""
        # TODO (STEP 8): residual + attention -> layer_norm; residual + ffn -> layer_norm
        raise NotImplementedError
