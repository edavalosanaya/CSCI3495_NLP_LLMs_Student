# HW3: Attention & the Transformer (Implement From Scratch)

**Out:** Week 6, Class 1 · **Due:** Week 7, Class 1 (start of class)
**100 points** · **Weight:** 2.5% of the course grade · **Individual assignment** · **Estimated time:** 6-8 hours

---

## Learning goals
By completing this homework you will be able to:
1. Implement **scaled dot-product attention** and explain why it is scaled by `1/√d_k`.
2. Build **multi-head self-attention** by splitting, attending, and recombining heads.
3. Construct **sinusoidal positional encodings** and explain why they are needed.
4. Assemble a complete **Transformer encoder block** (residual connections, layer norm, a
   position-wise feed-forward network) and apply a **causal mask** for autoregressive
   attention.

## Background
This assignment is the technical heart of the course. It corresponds to Week 5 (seq2seq &
attention) and Week 6 (pretraining), and to **Jurafsky & Martin Ch. 9-10**. The required
papers are:

- **Bahdanau, Cho & Bengio (2014)**, "Neural Machine Translation by Jointly Learning to
  Align and Translate", arXiv:1409.0473, introduced *additive attention* in seq2seq.
- **Vaswani et al. (2017)**, "Attention Is All You Need", arXiv:1706.03762, replaced
  recurrence entirely with *self-attention* and introduced the Transformer.

The Transformer's core operation is **scaled dot-product attention**:

> Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k ) V

The `1/√d_k` scaling keeps the dot products from growing large in magnitude (which would
push softmax into regions with vanishing gradients). **Multi-head attention** runs `h`
attention operations in parallel subspaces and concatenates them, letting the model attend
to different relations at once. Because attention is **permutation-invariant**, the model
has no inherent notion of order, so **positional encodings** are added to the inputs.

You will implement every piece in **NumPy only**, no `torch.nn`, no framework attention.
All tensors are tiny (sequence lengths and dimensions in the single digits) so the tests
run in well under a second on a CPU.

## Files

```
hw3/
  transformer.py        # <- YOU implement the TODOs here
  test_transformer.py   # the tests each step below refers to
  README.md             # this handout
```

## How this homework works

This handout is a sequence of steps. Each step is one function, and **each step
ends with a test you can run**, so you always know whether you are done before
you move on. Work them in order: later steps import earlier ones.

From the repository root, inside the course image:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps course \
    python -m pytest homeworks/hw3 -q
```

That is a mouthful to retype, so make a shortcut for the session:

```bash
alias hw='docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw3 -q'
```

Then:

```bash
hw -k step3      # check ONLY step 3
hw               # run every step
```

If you already work inside the container (`... run --rm --no-deps course bash`),
drop the docker prefix and just use `python -m pytest homeworks/hw3 -q`.

**Before you write anything, every test skips.** That is expected: the suite
detects the unfinished starter and skips rather than drowning you in failures.
The moment step 1 is implemented the tests start running for real.

**Total when you are finished: `22 passed`.**

### Step 0, Orientation (0 pts)

Nothing to write yet.

Read `transformer.py` top to bottom. Everything is NumPy: no PyTorch, no autograd,
no training. You are building the forward pass only, which is exactly the part you
will be asked to reason about on the exam. Then:

```bash
hw
```

You should get `22 skipped`.

### Step 1, `softmax` (5 pts)

**Write** `softmax(x, axis)`. Subtract the max along `axis` before exponentiating, then normalize. The subtraction changes nothing mathematically and everything numerically.

**Done when** `hw -k step1` prints `3 passed, 19 deselected`.

**Check it by hand**

```python
>>> import numpy as np
>>> from transformer import softmax
>>> np.round(softmax(np.array([0., 0., 0.])), 4)
array([0.3333, 0.3333, 0.3333])
>>> np.round(softmax(np.array([1., 2., 3.])), 4)
array([0.09  , 0.2447, 0.6652])
>>> softmax(np.array([1000., 1001.]))          # would be nan without the max trick
array([0.26894142, 0.73105858])
```

**Why it matters.** Every attention weight in the rest of this assignment comes out of this function. The max subtraction is the difference between a Transformer that runs on real logits and one that returns `nan` the first time a score exceeds about 710.

### Step 2, `scaled_dot_product_attention` (15 pts)

**Write** the attention function: `scores = Q @ K.T / sqrt(d_k)`, add the mask if one is given, softmax over the last axis, then multiply by `V`. Return both the output and the weights.

**Done when** `hw -k step2` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> from transformer import scaled_dot_product_attention
>>> Q = K = np.eye(2)
>>> V = np.array([[10., 0.], [0., 20.]])
>>> out, w = scaled_dot_product_attention(Q, K, V)
>>> np.round(w, 4)
array([[0.6698, 0.3302],
       [0.3302, 0.6698]])
>>> np.round(out, 4)
array([[ 6.6976,  6.6048],
       [ 3.3024, 13.3952]])
```

**Why it matters.** Read the weight matrix: each row sums to 1 and each position leans toward itself. The output is a weighted average of the value vectors, which is the one sentence worth memorizing about attention.

### Step 3, `causal_mask` (10 pts)

**Write** `causal_mask(seq_len)`: a `(seq_len, seq_len)` array that is 0 where a position may attend and a large negative number where it may not. Build it with `np.triu` or `np.tril` rather than a Python loop.

**Done when** `hw -k step3` prints `3 passed, 19 deselected`.

**Check it by hand**

```python
>>> from transformer import causal_mask
>>> causal_mask(3)
array([[-0.e+00, -1.e+09, -1.e+09],
       [-0.e+00, -0.e+00, -1.e+09],
       [-0.e+00, -0.e+00, -0.e+00]])
```

**Why it matters.** The mask is added to the scores *before* the softmax, which is why it is a large negative number and not a zero: `exp(-1e9)` underflows to 0, so the masked position gets exactly no weight. Multiplying by zero after the softmax would break the normalization instead.

### Step 4, `_split_heads` and `_combine_heads` (10 pts)

**Write** the two reshapes. `_split_heads` takes `(seq, d_model)` to `(num_heads, seq, d_head)`; `_combine_heads` is its exact inverse. Get the transpose right: reshape alone puts the axes in the wrong order.

**Done when** `hw -k step4` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> mha = ...  # a MultiHeadAttention with d_model=8, num_heads=2
>>> x = np.arange(3 * 8, dtype=float).reshape(3, 8)
>>> mha._split_heads(x).shape
(2, 3, 4)
>>> np.allclose(mha._combine_heads(mha._split_heads(x)), x)
True
```

**Why it matters.** Head 0 must get the first `d_head` channels of every position, head 1 the next block, and so on. If the transpose is wrong the shapes still work and the tests for later steps still run, but every head attends over a scrambled mixture of channels.

### Step 5, `MultiHeadAttention.forward` (20 pts)

**Write** `forward(x, mask)`: project `x` through `W_q`, `W_k`, `W_v`, split into heads, run step 2's attention per head (applying the mask to every head), combine, then project through `W_o`. Return the output and the per-head weights.

**Done when** `hw -k step5` prints `3 passed, 19 deselected`.

**Check it by hand**

```python
>>> out, w = mha.forward(x)                 # x is (seq, d_model)
>>> out.shape, w.shape                      # (seq, d_model), (num_heads, seq, seq)
((3, 8), (2, 3, 3))
>>> np.round(w.sum(axis=-1), 6)             # every row of every head sums to 1
array([[1., 1., 1.],
       [1., 1., 1.]])
```

**Why it matters.** With `num_heads=1` this must reduce exactly to step 2 with the projections applied. The test checks that, and it is the cleanest way to find a broken split or combine.

### Step 6, `positional_encoding` (10 pts)

**Write** `positional_encoding(seq_len, d_model)`: sine on the even channels, cosine on the odd ones, with the wavelength geometric in the channel index. Build the position column and the `div_term` and let broadcasting do the rest.

**Done when** `hw -k step6` prints `3 passed, 19 deselected`.

**Check it by hand**

```python
>>> from transformer import positional_encoding
>>> np.round(positional_encoding(1, 4)[0], 4)      # position 0: sin(0)=0, cos(0)=1
array([0., 1., 0., 1.])
>>> np.round(positional_encoding(2, 4)[1], 4)      # position 1
array([0.8415, 0.5403, 0.01  , 1.    ])
```

**Why it matters.** Position 0 is always `[0, 1, 0, 1, ...]`, which is a free check that your even/odd channel split is right. This exists because self-attention is permutation-equivariant: without it, 'dog bites man' and 'man bites dog' are the same input.

### Step 7, `layer_norm`, `relu`, `FeedForward.forward` (8 pts)

**Write** the three small pieces. `layer_norm` normalizes across the feature axis of each position (not across positions), then applies `gamma` and `beta`. `relu` is `maximum(x, 0)`. `FeedForward.forward` is `relu(x @ W1 + b1) @ W2 + b2`.

**Done when** `hw -k step7` prints `4 passed, 18 deselected`.

**Check it by hand**

```python
>>> from transformer import layer_norm, relu
>>> x = np.array([[1., 2., 3., 4.]])
>>> np.round(layer_norm(x, np.ones(4), np.zeros(4))[0], 4)
array([-1.3416, -0.4472,  0.4472,  1.3416])
>>> relu(np.array([-1., 0., 2.]))
array([0., 0., 2.])
```

**Why it matters.** Check that the normalized row has mean 0 and unit variance. Normalizing across the wrong axis is the classic bug here, and it produces plausible-looking numbers that make the encoder block silently wrong.

### Step 8, `EncoderBlock.forward` (7 pts)

**Write** the block: `x = layer_norm(x + attention(x))`, then `x = layer_norm(x + ffn(x))`. Two residual connections, two norms, shape preserved end to end.

**Done when** `hw -k step8` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> out = block.forward(x)          # x is (seq, d_model)
>>> out.shape == x.shape
True
>>> np.round(out.mean(axis=-1), 6)  # each position is layer-normed at the output
array([ 0., 0., 0.])
```

**Why it matters.** Shape preservation is what lets you stack this block N times, which is the entire architecture. The residual `x +` is what lets gradients reach the bottom of a deep stack; drop it and a 12-layer model stops training.

### Step 9, Run the whole thing (0 pts)

```bash
hw
```

Every step green means `22 passed`. If a step you finished earlier has gone red,
you broke it with a later change; fix that before you submit.

## Written reflection (15 pts)

Answer in the module docstring or a short `REFLECTION.md`, a paragraph each:

1. You divided by `sqrt(d_k)` in step 2. Work out what happens to the softmax if you
   remove it and `d_k` is 64, and say why that hurts training specifically.
2. Step 5 runs several heads in parallel instead of one wide attention. What does that
   buy, given the total parameter count is the same?
3. Your encoder block has no causal mask, but step 3 built one. Name a task that needs
   the mask and a task that is better off without it, and say why.

## What to submit

- `transformer.py` with every TODO filled in and `hw` fully green.
- Your reflection (in the module docstring or `REFLECTION.md`).
- The `AI-USE:` note described below.

Partial credit follows the tests: each step is worth the points listed above, and a
step whose tests pass earns them. Code that does not import earns at most the
reflection points, so submit something that runs even if it is incomplete.

## AI-use disclosure (required)

Per the syllabus, you may use LLM tools as coding assistants, but you must
**disclose** it (which tool, for what), be able to **explain every line** you
submit, and write the reflection in your own words. Put a short `AI-USE:` note
in your file header. Undisclosed AI use is an academic-integrity violation.
