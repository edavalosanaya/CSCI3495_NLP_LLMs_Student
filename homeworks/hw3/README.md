# HW3: Attention & the Transformer (Implement From Scratch)

**Out:** Week 6, Class 1 · **Due:** Week 7, Class 1 (start of class)
**Weight:** 5% of course grade · **Individual assignment** · **Estimated time:** 6-8 hours

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
  test_transformer.py   # public tests (known-value + shape checks)
  README.md             # this handout
```

## Tasks

### Task 1: Softmax (5 pts)
**`softmax(x, axis)`**, numerically stable (subtract the max before `exp`). Output sums to
1 along `axis`, and must not overflow on large logits.

### Task 2: Scaled dot-product attention (25 pts)
1. **`scaled_dot_product_attention(Q, K, V, mask)`**, return `(output, weights)`. Support
   an optional additive `mask` that broadcasts over leading dimensions. Weights are
   probabilities (rows sum to 1).
2. **`causal_mask(seq_len)`**, an additive mask that is `0` on/below the diagonal and a
   large negative number above it, so a token cannot attend to the future.

### Task 3: Multi-head attention (30 pts)
Complete `MultiHeadAttention`:
3. **`_split_heads` / `_combine_heads`**, reshape between `(seq, d_model)` and
   `(num_heads, seq, d_head)` (they must round-trip exactly).
4. **`forward(x, mask)`**, project with `W_q/W_k/W_v`, split into heads, run per-head
   scaled dot-product attention (the mask broadcasts over heads), recombine, and apply the
   output projection `W_o`. Return `(output, weights)` with weights shaped
   `(num_heads, seq, seq)`.

### Task 4: Positional encoding (10 pts)
5. **`positional_encoding(seq_len, d_model)`**, the sinusoidal scheme from Vaswani et al.,
   Eq. 3.5: even channels use `sin`, odd channels use `cos`, with the `10000^(2i/d_model)`
   wavelength schedule.

### Task 5: Encoder block (15 pts)
6. **`layer_norm(x, gamma, beta, eps)`**, normalize over the last (feature) axis, then
   apply the affine transform.
7. **`relu(x)`** and **`FeedForward.forward(x)`**, the position-wise FFN
   `ReLU(x W1 + b1) W2 + b2`.
8. **`EncoderBlock.forward(x, mask)`**, the post-LayerNorm block:
   `a = LayerNorm(x + MHA(x))`, then `out = LayerNorm(a + FFN(a))`.

### Written reflection (15 pts)
In a `REFLECTION.md` (≤ 250 words):
- **(a)** Why divide the attention logits by `√d_k`? What goes wrong without it?
- **(b)** Self-attention has no recurrence or convolution, what does the model lose, and
  how do positional encodings restore it?
- **(c)** Bahdanau attention is *additive*; Transformer attention is *(scaled) dot-product*.
  Give one practical reason the dot-product form was preferred at scale.

## Deliverables
- Completed `transformer.py` passing the public tests.
- `REFLECTION.md`.
- AI-use disclosure (see below).

## Grading rubric (100 pts)
| Component | Points |
|---|---:|
| Task 1: `softmax` (stable, normalized) | 5 |
| Task 2: scaled dot-product attention + `causal_mask` | 25 |
| Task 3: multi-head attention (split/combine + forward) | 30 |
| Task 4: sinusoidal positional encoding | 10 |
| Task 5: `layer_norm`, FFN, `EncoderBlock` | 15 |
| Written reflection | 15 |
| **Total** | **100** |

Partial credit is awarded per passing test.

## How to run & test
All code runs in the course Docker image (CPU-only, no network):

```bash
# Run the public tests against YOUR code:
docker compose -f docker/docker-compose.yml run --rm course \
    python -m pytest homeworks/hw3 -q

# (Instructor / self-check) against the reference solution:
docker compose -f docker/docker-compose.yml run --rm \
    -e HW3_FROM=solution course \
    python -m pytest homeworks/hw3 -q
```

Before you implement anything the suite **skips** (expected). Target: **20/20 passing**.
The tests include *known-value* checks (e.g. uniform attention returns the mean of `V`,
softmax of equal logits is uniform, positional encodings at `pos=0` are `sin=0`/`cos=1`),
so they validate correctness, not just shapes.

## AI-use disclosure (required)
Per the syllabus AI-use policy: **(a)** disclose any AI assistance in your file header
(which tool, for what), **(b)** be able to explain every line you submit, and **(c)** write
the reflection in your own words. Add an `AI-USE:` note in your header. Undisclosed AI use
is an academic-integrity violation.
