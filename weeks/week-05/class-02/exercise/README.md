# W5C2 Lab: Build the Attention Block

## 1. Learning objective

Implement the Transformer's attention from scratch in NumPy: the scaled
dot-product, the causal mask that hides the future, and the multi-head wrapper.

You write three functions in `attention_lab.py`. The softmax and the
head split/combine helpers are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-05/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `scaled_dot_product_attention`

![The Transformer's attention in one formula](../lecture/visuals/scaled-dot-formula.png)

Queries meet keys, the scores are scaled by $\sqrt{d_k}$ and normalized, and
the result blends the values:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The $\sqrt{d_k}$ is not decoration. Dot products grow with dimension, and a
softmax over large numbers collapses onto one position and stops passing
gradients back.

Transpose only the last two axes, scale before the softmax, and send masked
positions to a large negative number rather than deleting them.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 4 deselected
```

## 4. Implement `causal_mask`

One line: a `(T, T)` boolean whose lower triangle, diagonal included, is True.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 4 deselected
```

## 5. Implement `multi_head_attention`

![Vaswani et al. 2017, Figure 2: scaled dot-product attention (left) and multi-head attention (right)](../lecture/visuals/assets/vaswani-2017-fig-2.png)

Each head runs that same attention on its own projection of $Q$, $K$ and $V$,
and the heads are concatenated and mixed by one more matrix:

$$\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V) \qquad \mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W^O$$

Project, split, attend, combine, mix. Every piece already exists.

```bash
pytest -k step3 -q
```

```
.                                                                        [100%]
1 passed, 4 deselected
```

## 6. Run it, then break it

```bash
python attention_lab.py
```

```
MILESTONE 1  scaled dot-product attention: WORKS
  weights row 0: [0.598 0.104 0.2   0.097]  (sums to 1.0 )
MILESTONE 2  causal mask: WORKS
  masked weights (rounded); note the zeros ABOVE the diagonal, the future:
    [1. 0. 0. 0.]
    [0.02 0.98 0.   0.  ]
    [0.17 0.22 0.62 0.  ]
    [0.08 0.3  0.11 0.51]
MILESTONE 3  multi-head attention: WORKS, output shape (4, 8)
```

Each experiment below is a one-line edit; undo it before the next.

1. Drop the $\sqrt{d_k}$. Remove the division and re-run. Row 0 goes from
   `[0.598 0.104 0.2 0.097]` to `[0.945 0.007 0.043 0.006]`: one position now
   takes almost everything. Explain what that does to the gradient reaching the
   other three positions.
2. Mask with zero instead of $-10^9$. Replace the masked score with `0.0`. Row
   0 goes from `[1. 0. 0. 0.]` to `[0.532 0.156 0.156 0.156]`: the model is now
   reading three tokens from the future. Zero is exactly the WEIGHT you want,
   so why is it the wrong SCORE?
3. Change the head count. Call `multi_head_attention` with `num_heads` of 1, 2
   and 4. The output is `(4, 8)` every time. If the shape never changes, what
   is actually different about what the block computed?
4. Look at row 0 of the masked weights: `[1. 0. 0. 0.]`. The first token
   attends only to itself, with no choice in the matter. What does that imply
   about how much the first position of a sequence can ever contribute?
