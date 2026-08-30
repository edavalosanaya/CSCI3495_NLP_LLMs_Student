# W5C1 Walkthrough: Additive attention, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `attention.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it at `SEED = 0`.

---

## Step 0, Orientation

The module holds three learned tensors:

```python
        self.W_s = nn.Parameter(torch.randn(attn, hidden) * 0.1)
        self.W_h = nn.Parameter(torch.randn(attn, hidden) * 0.1)
        self.v = nn.Parameter(torch.randn(attn) * 0.1)
```

`W_s` and `W_h` lift the query and each key from `hidden = 8` into a 16-dim
attention space; `v` collapses that space to one scalar. So the "scorer" is a
one-hidden-layer MLP with a `tanh`, and the whole of attention's learnable
capacity lives in those three tensors.

The `* 0.1` initialization matters for the demo: small weights mean small
scores, and softmax of small scores is nearly uniform. That is what produces the
gray blur students see first, and it is a deliberate teaching setup rather than
an accident.

---

## Step 1, Score every key

```python
    sq = query @ W_s.T              # (attn,)
    kh = keys @ W_h.T              # (num_keys, attn)
    pre = torch.tanh(sq + kh)     # (num_keys, attn) via broadcast
    return pre @ v                # (num_keys,)
```

**The broadcast is the only subtle line.** `sq` has shape `(16,)` and `kh` has
shape `(3, 16)`. PyTorch broadcasts `sq` across all three rows, so one expression
computes $W_s s + W_h h_i$ for every $i$. Writing this as a Python loop over keys
gives identical numbers and is a perfectly good first draft; the vectorized form
is what makes attention affordable at sequence length 2000.

**Why `.T`.** `W_s` is stored `(attn, hidden)` to match the math ($W_s s$ with $s$
a column vector). In PyTorch we carry row vectors, so the multiplication needs
the transpose. A student who defines the parameters as `(hidden, attn)` and drops
the `.T` gets the same result; the tests only check shapes and behavior.

**Why `tanh` before `v`.** Without a nonlinearity, $v^\top(W_s s + W_h h_i)$
collapses to a single linear function of $s$ and $h_i$, and the score could not
represent interactions between them. The `tanh` is what makes this an *additive*
(as opposed to bilinear) scorer, and it is Bahdanau et al.'s original choice.

**What you should see:**

```python
>>> e = additive_scores(torch.eye(3, 8)[1], keys, attn.W_s, attn.W_h, attn.v)
>>> tuple(e.shape)
(3,)
```

---

## Step 2, Softmax and blend

```python
        e = additive_scores(query, keys, self.W_s, self.W_h, self.v)
        weights = torch.softmax(e, dim=0)
        context = weights @ values
        return context, weights
```

**`dim=0` normalizes across keys.** With `e` of shape `(num_keys,)` there is only
one axis, so this is unambiguous here, but the habit matters: in the batched
multi-head version next class, softmaxing the wrong axis is a real and silent
bug, because the output still sums to 1 along *some* axis.

**`weights @ values` is a weighted sum.** With `weights` shape `(3,)` and
`values` shape `(3, 8)`, the matmul contracts the key axis and returns `(8,)`.
This is $c_t = \sum_i \alpha_{t,i} h_i$ written without a loop.

**Note there are no parameters after the scores.** Softmax and the weighted sum
are fixed operations. All of attention's learning capacity is in Step 1, which is
worth saying because students often imagine the "attention weights" are
parameters. They are activations, recomputed for every query.

**What you should see:**

```python
>>> context, weights = attn(torch.eye(3, 8)[1], keys, keys)
>>> [round(float(w), 3) for w in weights]
[0.328, 0.322, 0.35]
>>> round(float(weights.sum()), 4)
1.0
```

**Roughly 1/3 each is the correct output, not a bug**, and it is worth stopping
on. The scorer is randomly initialized with small weights, so all three scores are
near zero and softmax of near-equal scores is near-uniform. A student who
"fixes" this by tweaking initialization has misunderstood: the flat distribution
is the honest state of an untrained model.

---

## Step 3, Draw the heatmap

```python
        for c in range(cols):
            level = int(round(float(W[r, c]) * (len(ramp) - 1)))
            level = max(0, min(len(ramp) - 1, level))
            cells.append(ramp[level] * 4)
```

**The clamp is load-bearing.** `ramp` has 10 characters, so valid indices are 0
to 9. A weight of exactly 1.0 gives `round(1.0 * 9) = 9`, fine, but floating
point can produce a weight a hair above 1.0 after softmax, and `ramp[10]` raises
`IndexError` in the middle of a demo. Clamping costs one line.

**`* 4`** makes each cell four characters wide so the grid lines up with the
4-wide column headers. Purely cosmetic, but the test checks the rendered grid
shape.

---

## Step 4, Run the whole thing

```
UNTRAINED attention weights: [0.328, 0.322, 0.35]
weights sum: 1.0

Untrained heatmap (3 queries x 3 keys), a uniform gray blur:
        k0   k1   k2
   q0 ---- ---- ----
   q1 ---- ---- ----
   q2 ---- ---- ----

TRAINED attention weights (query 1): [0.0, 1.0, 0.0]
context vector: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Trained heatmap, the diagonal emerges (dark = high weight):
        k0   k1   k2
   q0 @@@@
   q1      @@@@
   q2           @@@@
```

**Teach the pair of heatmaps, not just the second one.** This is the highest-value
five minutes in the session.

`align_briefly` runs 300 Adam steps on the objective $-\sum_i \log \alpha_{ii}$,
that is, "reward query $i$ for putting its mass on key $i$". Nothing else changed:
same architecture, same keys, same values. The flat field became a crisp diagonal
purely because a gradient told it to.

The point to land: **an attention heatmap is a picture of what the model was
trained to do, not evidence that the mechanism understands an alignment.** The
literature on attention-as-explanation (Jain and Wallace 2019, "Attention is not
Explanation"; Wiegreffe and Pinter 2019 in reply) is exactly this argument at
scale, and a student who has watched the diagonal appear on command is well
placed to follow it.

Two useful follow-ups you can run live:

- Change `align_briefly` to reward $\alpha_{i,i+1}$ and the diagonal shifts one
  column. The mechanism learns whatever objective it is given.
- Comment out the `align_briefly` call and the heatmap stays gray forever, no
  matter how many times you re-run. The structure does not emerge from the
  architecture.

**On the context vector.** After training, query 1's context is
`[0, 1, 0, 0, 0, 0, 0, 0]`, exactly `keys[1]`, because attention collapsed onto a
single key and `values = keys`. In a real encoder-decoder the weights stay
diffuse and the context is a genuine blend, which is the property that fixed the
fixed-length-bottleneck problem from W4C2.
