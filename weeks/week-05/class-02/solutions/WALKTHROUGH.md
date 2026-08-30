# W5C2 Walkthrough: Self-attention from scratch, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `attention_lab.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it.

This is the full-period lab, so students will be at very different steps at any
moment. The progressive runner in `main()` is designed for exactly that: it
reports the last milestone that works, so a student can call you over and you can
see their position in one command.

---

## Step 1, Numerically stable softmax

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
```

**Why subtracting the max is mathematically free.** Softmax is invariant to a
constant shift: $e^{x_i - c} / \sum_j e^{x_j - c} = e^{x_i} / \sum_j e^{x_j}$
because $e^{-c}$ cancels top and bottom. So we may choose any $c$, and choosing
$c = \max x$ guarantees every exponent is at most 0, hence every $e^{x}$ is at
most 1. No overflow, ever.

**The demonstration to run live:**

```python
>>> softmax(np.array([1.0, 2.0, 3.0]))
array([0.09003057, 0.24472847, 0.66524096])
>>> softmax(np.array([1000.0, 1001.0, 1002.0]))
array([0.09003057, 0.24472847, 0.66524096])
```

Identical, as the algebra promises. The naive version returns `nan` for the
second input, since `np.exp(1000)` is `inf` and `inf/inf` is undefined. Students
who think this is a contrived case should be reminded that attention scores scale
with $d_k$, and production models run $d_k$ in the hundreds.

**`keepdims=True` in both calls** preserves the reduced axis as length 1 so the
subtraction and division broadcast back over the original shape. Without it,
NumPy drops the axis and the broadcast either errors or, worse, silently
broadcasts along the wrong axis.

---

## Step 2, Scaled dot-product attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)  # (...,Tq,Tk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
```

**`np.swapaxes(K, -1, -2)` rather than `K.T`.** This is the single most
consequential line in the file. In Step 5 the same function is called with 3-D
arrays shaped `(num_heads, T, d_head)`. `K.T` reverses *every* axis, producing
`(d_head, T, num_heads)`, which then either raises a shape error or, depending on
the dimensions chosen, quietly multiplies the wrong things. `swapaxes(-1, -2)`
transposes only the last two axes and therefore works unchanged for 2-D and 3-D.
Students who used `.T` will pass Steps 2 and 3 and fail Step 5 with a confusing
error; point them here immediately.

**Why divide by $\sqrt{d_k}$.** If the components of $q$ and $k$ are independent
with mean 0 and variance 1, then $q \cdot k$ has variance $d_k$. As $d_k$ grows
the scores spread out, softmax approaches a one-hot, and the gradient through it
approaches zero. Dividing by $\sqrt{d_k}$ restores unit variance. This is
Vaswani et al.'s footnote 4, and it is the reason the mechanism is called
*scaled* dot-product attention.

**Returning weights as well as output** costs nothing and is what makes the
heatmaps and the masked-weight printout possible.

**What you should see:**

```
MILESTONE 1  softmax + scaled dot-product attention: WORKS
  weights row 0: [0.598 0.104 0.2   0.097]  (sums to 1.0 )
```

---

## Step 3, The causal mask

```python
def causal_mask(T: int) -> np.ndarray:
    """Lower-triangular True mask: position t may attend to <= t."""
    return np.tril(np.ones((T, T), dtype=bool))
```

**Where the mask is applied is the conceptual point.** It goes on the *scores*,
before the softmax:

```python
        scores = np.where(mask, scores, -1e9)
```

A masked score of $-10^9$ exponentiates to (numerically) 0, so its weight is
exactly zero **and the surviving weights renormalize to sum to 1**. If instead
you softmaxed first and then zeroed the masked weights, the rows would no longer
sum to 1 and the context vectors would be systematically shrunk toward zero for
early positions. This is a real bug that produces plausible-looking training
curves, so it is worth making students say out loud why the order matters.

Some implementations use `-np.inf` instead of `-1e9`. That also works here, but
`-inf` can produce `nan` when an entire row is masked (as happens with padding
masks), because then softmax computes `0/0`. `-1e9` degrades to a uniform
distribution instead of `nan`, which is why most production code uses a large
finite number.

**What you should see:**

```
MILESTONE 2  causal mask: WORKS
  masked weights (rounded); note the zeros ABOVE the diagonal, the future:
    [1. 0. 0. 0.]
    [0.02 0.98 0.   0.  ]
    [0.17 0.22 0.62 0.  ]
    [0.08 0.3  0.11 0.51]
```

**Row 0 is `[1, 0, 0, 0]` exactly.** The first token has exactly one legal
position to attend to (itself), and softmax over a single value is 1.0 regardless
of the score. Nice concrete evidence that the mask really is total.

**The payoff line for the lecture:** this triangle is the entire architectural
difference between BERT and GPT. Remove it and the model sees the future, which
makes next-token prediction trivial and useless. Keep it and the same code is a
causal language model.

---

## Step 4, Split and combine heads

```python
def split_heads(X, num_heads):
    """(T, d_model) -> (num_heads, T, d_head)."""
    T, d_model = X.shape
    d_head = d_model // num_heads
    return X.reshape(T, num_heads, d_head).transpose(1, 0, 2)


def combine_heads(X):
    """(num_heads, T, d_head) -> (T, num_heads*d_head)."""
    num_heads, T, d_head = X.shape
    return X.transpose(1, 0, 2).reshape(T, num_heads * d_head)
```

**Two operations, and both are needed.** The `reshape` slices the feature axis
into `num_heads` contiguous blocks, giving `(T, num_heads, d_head)`. The
`transpose(1, 0, 2)` then moves the head axis to the front, giving
`(num_heads, T, d_head)`, which is what lets `scaled_dot_product_attention` treat
heads as a batch dimension and process them all in one matmul.

**Why the test is a round trip.** A student who forgets the transpose gets shapes
that look reasonable and numbers that are wrong, because the reshape in
`combine_heads` would then interleave features from different heads.
`combine_heads(split_heads(X)) == X` catches this exactly; a shape-only assertion
would not.

**What you should see:**

```python
>>> X = np.arange(24).reshape(4, 6)
>>> split_heads(X, 2).shape
(2, 4, 3)
>>> np.array_equal(combine_heads(split_heads(X, 2)), X)
True
```

**The conceptual framing worth giving:** multi-head attention is not a new
mechanism. It is the *same* attention run on disjoint slices of the feature
dimension, so each head can specialize, followed by a projection that lets the
model mix what the heads found. Note that `d_head = d_model // num_heads`, so
more heads means narrower heads: the total compute is roughly constant.

---

## Step 5, Multi-head attention

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """X:(T,d_model); W*:(d_model,d_model). Returns (T, d_model)."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    Qh, Kh, Vh = (split_heads(M, num_heads) for M in (Q, K, V))
    out_h, _ = scaled_dot_product_attention(Qh, Kh, Vh, mask=mask)
    return combine_heads(out_h) @ Wo
```

**Four lines, one per stage of the right-hand diagram in Vaswani Fig. 2:**
project, split, attend, combine-and-project.

**`scaled_dot_product_attention` is called once, not in a loop.** The head axis
rides along as a batch dimension through every matmul and through the softmax
(which normalizes `axis=-1`, the key axis, unaffected by the leading head axis).
This is the whole reason attention is efficient on parallel hardware, and it is
worth pointing at explicitly: the naive mental model is "run attention h times",
and the code shows why nobody does that.

**The mask broadcasts.** `mask` is `(T, T)` while the scores are
`(num_heads, T, T)`. NumPy broadcasts the mask across heads automatically, so
every head gets the same causal constraint without any extra code.

**`X @ Wq` where `Wq` is `(d_model, d_model)`.** In the paper each head has its
own smaller projection $W_i^Q$ of shape `(d_model, d_head)`. Concatenating those
`h` matrices side by side gives exactly one `(d_model, d_model)` matrix, so a
single projection followed by `split_heads` is mathematically identical and
faster. Worth mentioning, because students comparing the code to the paper often
think a step is missing.

**What you should see:**

```
MILESTONE 3  multi-head attention: WORKS, output shape (4, 8)
```

Input `(4, 8)`, output `(4, 8)`. Attention is shape-preserving, which is what lets
you stack it into deep layers.

---

## Step 6, The checkpoint

```
.....                                                                    [100%]
5 passed
```

**The final test earns its keep:** it asserts that `multi_head_attention` with
`num_heads=1` matches plain `scaled_dot_product_attention` (modulo `Wo`). That is
a genuine correctness check rather than a shape check, and it fails for the
common transpose and reshape mistakes that every other test lets through.

**Closing the lab.** Forty lines of NumPy, and it is the actual computation
inside every Transformer in the rest of the course. What a production
implementation adds is engineering, not concept: batching, layer normalization,
residual connections, a position-wise feed-forward block, dropout, and a great
deal of kernel optimization. The attention is what students just wrote.

**If you have time at the end**, the highest-value five minutes is removing the
$\sqrt{d_k}$ and re-running with `d = 64`. The weight rows collapse to nearly
one-hot, which makes the scaling term's purpose concrete rather than a fact to
memorize.
