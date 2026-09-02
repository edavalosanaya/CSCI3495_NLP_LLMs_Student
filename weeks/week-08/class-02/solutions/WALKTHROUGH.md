# W8C2 Walkthrough: Be the reward model, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `preferences.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it.

---

## Orientation

```python
PREFERENCES = [("A","B"), ("A","C"), ("A","D"), ("B","C"), ("B","D"), ("C","D")]
```

Six pairs, complete and consistent, encoding `A > B > C > D`. **There are no
scores anywhere in the input.** That is not a simplification, it is the actual
design of RLHF Stage 2: humans are reasonably consistent at "which of these two
is better" and badly inconsistent at "rate this 1 to 10", so the data collected is
comparisons and the scalar is inferred.

---

## Given, `sigmoid`

```python
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
```

**Both branches are the same function**, algebraically. Multiply
$1/(1+e^{-x})$ top and bottom by $e^{x}$ and you get $e^{x}/(1+e^{x})$. The
point is that each branch only ever exponentiates a **negative** number, so the
result is in $(0, 1]$ and cannot overflow.

The naive single-expression version raises `OverflowError` for `x` below about
-745 in float64. Score differences in a reward model routinely reach that range
once training has separated the extremes, so this is not hypothetical.

```python
>>> sigmoid(-800.0)
0.0
```

---

## Step 1, `neg_log_likelihood`

```python
    total = 0.0
    for w, l in prefs:
        p = sigmoid(scores[w] - scores[l])
        # Clamp to avoid log(0).
        p = min(max(p, 1e-12), 1.0)
        total += -math.log(p)
    return total / len(prefs)
```

**Only the difference matters**, `scores[w] - scores[l]`. This is the property
that makes the scores unidentifiable up to a constant, which Step 3 has to handle.

**The clamp earns its place.** A confidently wrong ordering drives `p` to
underflow to exactly 0.0, and `math.log(0)` raises `ValueError`. Clamping turns
"infinitely surprised" into "very surprised", which keeps the loop running. Real
implementations use the log-sigmoid function directly, which is stable without a
clamp, and that is a reasonable thing to mention.

**The mean, not the sum**, so the loss is comparable across datasets of different
sizes and the learning rate does not have to be rescaled when preferences are
added.

**What you should see:**

```python
>>> flat = {k: 0.0 for k in "ABCD"}
>>> round(neg_log_likelihood(flat, PREFERENCES), 4)
0.6931
>>> good = {"A": 3.0, "B": 1.0, "C": -1.0, "D": -3.0}
>>> round(neg_log_likelihood(good, PREFERENCES), 4)
0.0699
```

**0.6931 is $\ln 2$.** With all scores equal, every comparison is 50/50 and the
loss is the entropy of a fair coin. Students have now seen this number as the
starting loss in W4C1 (binary classifier), W4C2 (with $\ln 23$ for 23 characters),
and here. Worth naming the pattern: an uninformed model's loss is the entropy of
its uninformed guess, and checking that is how you know your loss function is
wired correctly.

---

## Step 2, `fit_reward_model`

```python
    items = sorted({x for pair in prefs for x in pair})
    scores = {x: 0.0 for x in items}
    for _ in range(steps):
        grad = {x: 0.0 for x in items}
        for w, l in prefs:
            # d/ds of -log sigmoid(s_w - s_l):  s_w gets +(1-p), s_l gets -(1-p)
            p = sigmoid(scores[w] - scores[l])
            push = 1.0 - p
            grad[w] += push
            grad[l] -= push
        for x in items:
            scores[x] += lr * grad[x] / len(prefs)
        # Re-center: scores are only identifiable up to an additive constant.
        mean = sum(scores.values()) / len(scores)
        for x in items:
            scores[x] -= mean
    return scores
```

**`push = 1 - p` is the entire learning signal**, and it is worth deriving on the
board because it is unusually clean. The loss for one pair is
$-\log \sigma(s_w - s_l)$, and

$$\frac{\partial}{\partial s_w}\Big[-\log \sigma(s_w - s_l)\Big] = -(1 - \sigma(s_w - s_l))$$

so gradient *descent* moves $s_w$ up by $(1 - p)$ and $s_l$ down by the same
amount. Read what that means behaviorally:

- The model already believes the winner wins (`p` near 1): `push` near 0, almost
  no update. Settled preferences stop teaching.
- The model has it backwards (`p` near 0): `push` near 1, maximum correction.
  Violated preferences dominate learning.

This self-balancing is not designed in; it falls out of the likelihood. Compare
with the hinge-style losses students may have seen, which need an explicit margin.

**`sorted(...)` on the item set** makes the iteration order deterministic. Set
iteration order varies between runs, and floating-point addition is not
associative, so without the sort two runs can differ in the last decimal place.
Minor, but it is the same determinism discipline as the tie-breaks in W2C1 and
W8C1.

**Re-centering is the conceptually important line.** Since only differences enter
the loss, adding 100 to every score changes nothing observable. The scores live
in a one-dimensional family of equivalent solutions, and gradient descent will
happily drift along it. Re-centering to mean 0 picks one representative. There is
a test asserting this (`test_step3_scores_centered`), and the third stretch goal
(fit a single preference) makes the under-determination vivid: two items, one
comparison, and infinitely many score pairs fit equally well.

---

## Running it

```
Learned reward-model scores (higher = more preferred):
  A: +5.010
  B: +1.579
  C: -1.579
  D: -5.010
Implied ranking: A > B > C > D
```

**The ranking is recovered exactly from comparisons alone.** No score was ever
supplied. That is Stage 2 of RLHF, and students have now done it by hand.

**The uneven spacing is the part worth teaching.** The A-to-B gap is 3.43; the
B-to-C gap is 3.16. The scores are symmetric about zero because of re-centering,
but the extremes are pushed further out: A appears only as a winner and D only as
a loser, so their gradients never receive an opposing push, while B and C are
pulled from both sides.

Draw the conclusion explicitly: **the ordering is meaningful, the magnitudes are
much less so.** A reward model's scalar is not a calibrated quality rating. This
matters directly for Stage 3, where PPO optimizes *against* these numbers: a
policy that discovers a region where the reward model scores absurdly high will
exploit it, whether or not the responses are actually good. That is reward
hacking, and the seed of it is visible right here in the unbounded extremes.

**Then have them use their own labels** from `label_sheet.md`. The moment a
student's own inconsistent rankings produce compromise scores, the abstraction
becomes personal: the model learned *their* values, disagreements and all. Ask
whose values a real reward model encodes, and note that InstructGPT's own paper
acknowledges its labelers were a small, non-representative group. That is the
limitation question on Quiz 8.

**Running the contradictory-preference stretch goal live is worth the two
minutes.** Add both `("A","B")` and `("B","A")` and the two scores collapse
toward each other, because the pushes cancel. Real preference datasets are full of
this, and the reward model does not resolve the disagreement, it averages it.
