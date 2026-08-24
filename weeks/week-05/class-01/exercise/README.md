# W5C1 Lab: Implement Additive (Bahdanau) Attention

**Goal:** implement **additive attention** from scratch in PyTorch, then
**visualize** the attention-weight matrix as a text heatmap. You will see the
three-step recipe (**score, softmax, blend**) turn into a few lines of tensor
code.

Everything runs **CPU-only** in a couple of seconds.

## Before you code: the picture and the math

![Additive attention math: score, weights, context](../lecture/visuals/additive-math.png)

These three lines from the lecture are the entire exercise. For decoder query $s_{t-1}$
and encoder states $h_i$:

$$e_{t,i} = v^\top \tanh(W_s s_{t-1} + W_h h_i) \qquad \alpha_{t,i} = \mathrm{softmax}_i(e_{t,i}) \qquad c_t = \sum_i \alpha_{t,i} h_i$$

![Attention alignment heatmap for "la maison bleue est grande"](../lecture/visuals/attention-alignment.png)

Your finished code takes one query and a stack of encoder states, scores each state with the tiny $\tanh$ MLP (that is `additive_scores`), softmaxes the scores into weights $\alpha_{t,i}$ that sum to 1, and returns the weighted blend $c_t$ plus the weights. The weights are what the heatmap above shades: your `heatmap` function draws the same picture in ASCII, one row per query. **Check yourself before coding:** if all scores $e_{t,i}$ come out equal, what does $c_t$ become? (The plain average of the $h_i$, because softmax turns equal scores into equal weights $1/n$, exactly the "uniform gray blur" you will see from the untrained scorer.)

> **Trace it by hand first (pairs, whiteboard, 15 min).** Use the worksheet
> numbers from the activity slide: keys = values h1 = [1, 0], h2 = [0, 1],
> h3 = [1, 1] and query q = [2, 0]. Compute the 3 dot-product **scores**
> (2, 0, 2), exponentiate with the supplied table (e^2 = 7.4, e^0 = 1.0),
> **softmax** into weights (0.47, 0.06, 0.47), take the **weighted blend** of the
> values (c = [0.94, 0.53]), and **shade** the 1x3 alignment bar. The steps below
> implement *exactly* what you traced.

## How this lab works

Each step tells you **what to write**, then exactly **how to check it**. Step 2
calls Step 1; Step 3 is independent, so you can do it first if the tensor algebra
is fighting you.

Set a shortcut for the long docker command first:

```bash
alias lab='docker compose -f docker/docker-compose.yml run --rm --no-deps course'
```

Check **one step**:

```bash
lab python -m pytest weeks/week-05/class-01/exercise/test_attention.py -k step1 -q
```

Check **everything**:

```bash
lab python -m pytest weeks/week-05/class-01/exercise/test_attention.py -q
```

Stuck for more than a few minutes? Open the walkthrough released after class at the
matching step.

---

### Step 0, Orientation (nothing to write)

Run the starter and confirm it stops at the first TODO:

```bash
lab python weeks/week-05/class-01/exercise/attention.py
```

Then look at the shapes you will be working with:

```bash
lab python
```

```python
>>> import sys, torch; sys.path.insert(0, "weeks/week-05/class-01/exercise")
>>> from attention import AdditiveAttention
>>> torch.manual_seed(0)
>>> keys = torch.eye(3, 8)      # three encoder states, 8-dimensional
>>> tuple(keys.shape)
(3, 8)
>>> attn = AdditiveAttention(hidden=8)
>>> tuple(attn.W_s.shape), tuple(attn.W_h.shape), tuple(attn.v.shape)
((16, 8), (16, 8), (16,))
```

**Notice:** `W_s` and `W_h` both map an 8-dimensional state into a 16-dimensional
"attention space", and `v` collapses that back down to a single number. That
single number is the score for one key. This little `tanh` sandwich is the entire
scoring network.

---

### Step 1, Score every key

**Write:** `additive_scores`, returning one score per key, shape `(num_keys,)`.

```
sq  = query @ W_s.T          # (attn,)
kh  = keys  @ W_h.T          # (num_keys, attn)
pre = torch.tanh(sq + kh)    # broadcast -> (num_keys, attn)
return pre @ v               # (num_keys,)
```

**The broadcast in line 3 is the trick.** `sq` is one vector of length `attn`;
`kh` is one such vector per key. Adding them broadcasts `sq` across every row, so
you compute $W_s s + W_h h_i$ for all $i$ at once instead of looping. That is the
same $W_s s_{t-1} + W_h h_i$ from the formula, vectorized.

**Done when:**

```bash
lab python -m pytest weeks/week-05/class-01/exercise/test_attention.py -k step1 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

**Check it by hand:**

```python
>>> from attention import additive_scores
>>> e = additive_scores(torch.eye(3, 8)[1], keys, attn.W_s, attn.W_h, attn.v)
>>> tuple(e.shape)
(3,)
```

Three keys in, three scores out. One number per key, which is all a score is.

**Why it matters:** this is the only learned part of attention. Everything after
it (softmax, weighted sum) has no parameters at all.

---

### Step 2, Softmax and blend

**Write:** `AdditiveAttention.forward`. Three lines:

```
e = additive_scores(query, keys, self.W_s, self.W_h, self.v)
weights = torch.softmax(e, dim=0)
context = weights @ values
return context, weights
```

**`dim=0` matters.** You are normalizing across the keys, so the weights answer
"how much of my attention goes to each encoder state?" and sum to 1. Softmaxing
the wrong axis gives numbers that still look like probabilities but mean nothing.

**Done when:**

```bash
lab python -m pytest weeks/week-05/class-01/exercise/test_attention.py -k step2 -q
```

```
..                                                                       [100%]
2 passed, 2 deselected
```

**Check it by hand:**

```python
>>> context, weights = attn(torch.eye(3, 8)[1], keys, keys)
>>> [round(float(w), 3) for w in weights]
[0.328, 0.322, 0.35]
>>> round(float(weights.sum()), 4)
1.0
```

**All three weights are about 1/3, and that is the correct answer here.** The
scorer is randomly initialized, so it has no opinion about which key matters.
This is the "uniform gray blur" from the check-yourself question above. Attention
weights are *learned*, not built in.

**Why it matters:** `context = weights @ values` is the payoff. Instead of
squeezing a whole sequence into one fixed vector (the W4C2 bottleneck), the
decoder gets a *different* blend at every step, chosen by the scores.

---

### Step 3, Draw the heatmap

**Write:** the inner cell loop in `heatmap`. For each weight in `[0, 1]`, map it
to an index into `ramp = " .:-=+*#%@"` (10 levels, light to dark) and append a
**4-character-wide** cell of that shade.

```
level = int(round(float(W[r, c]) * (len(ramp) - 1)))
level = max(0, min(len(ramp) - 1, level))
cells.append(ramp[level] * 4)
```

The clamp is not optional: rounding can push the index to 10 and `ramp[10]`
raises `IndexError`.

**Done when:**

```bash
lab python -m pytest weeks/week-05/class-01/exercise/test_attention.py -k step3 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

**Why it matters:** attention weights are the field's favourite interpretability
picture, and drawing one yourself makes clear how much interpretation is being
done by the color ramp rather than by the model.

---

### Step 4, Run the whole thing

```bash
lab python weeks/week-05/class-01/exercise/attention.py
```

```
UNTRAINED attention weights: [0.328, 0.322, 0.35]
  (a random scorer has no opinion yet: everything gets ~1/3)
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

And the full suite:

```bash
lab python -m pytest weeks/week-05/class-01/exercise/test_attention.py -q
```

```
....                                                                     [100%]
4 passed
```

**The two heatmaps are the lesson.** Before training, every query attends
equally to every key: a flat gray field, no alignment. After 300 tiny gradient
steps in `align_briefly`, a crisp diagonal appears and query 1's weights are
`[0.0, 1.0, 0.0]`.

Nothing about the architecture put that diagonal there. It came from the training
objective, which rewarded query $i$ for attending to key $i$. When you read a
published attention heatmap that looks meaningfully aligned, that alignment is a
consequence of what the model was trained to do, not evidence that the mechanism
"understands" the correspondence.

## Stretch goals

- Replace additive scoring with **scaled dot-product**,
  $e_i = (s \cdot h_i)/\sqrt{d}$, and confirm the trained heatmap looks the same.
  That is the Week 5 Class 2 mechanism, with no parameters in the scorer at all.
- Drop the $\sqrt{d}$ and scale the keys up by 10. Watch the softmax saturate into
  a one-hot and the gradients vanish. That is exactly why the scaling term exists.
- Change `align_briefly` to reward query $i$ for attending to key $i+1$ and
  confirm the diagonal shifts. The mechanism learns whatever you ask of it.

A full reference solution is in the material released after class, and the step-by-step
explanation is in the walkthrough released after class (don't peek until you've tried).
