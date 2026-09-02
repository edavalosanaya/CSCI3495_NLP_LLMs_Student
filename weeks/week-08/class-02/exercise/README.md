# W8C2 Lab: Reward Model from Preferences

## 1. Learning objective

Learn a scalar reward from nothing but pairwise human choices, which is stage 2
of RLHF. No ratings, no labels, just "this one was better".

You write two functions in `preferences.py`. The numerically stable sigmoid is
given.

## 2. Understanding the math

![RLHF pipeline: SFT, then reward model, then RL optimization with PPO](../lecture/visuals/rlhf-pipeline.png)

The Bradley-Terry model says the chance a human prefers $w$ over $l$ depends
only on the DIFFERENCE of their scores:

$$P(w \succ l) \;=\; \sigma(s_w - s_l), \qquad \sigma(x) = \frac{1}{1 + e^{-x}},$$

so fitting the model means minimizing the negative log-likelihood of every
choice the humans actually made:

$$\mathcal{L} \;=\; -\frac{1}{N} \sum_{(w,\,l)} \log \sigma(s_w - s_l).$$

![Reward model: humans rank responses, a model learns a scalar score with a pairwise ranking loss](../lecture/visuals/stage2-rm.png)

Because only differences appear, the scores are pinned down only up to an
additive constant, which is why they get re-centred after every pass.

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-08/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `neg_log_likelihood`

Average $-\log \sigma(s_w - s_l)$ over the pairs, clamping the probability away
from 0 first.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 5 deselected
```

## 5. Implement `fit_reward_model`

Hand-written gradient descent: push winners up, losers down, then re-centre.

```bash
pytest -k step2 -q
```

```
...                                                                      [100%]
3 passed, 3 deselected
```

## 6. Run it, then break it

```bash
python preferences.py
```

```
Learned reward-model scores (higher = more preferred):
  A: +5.010
  B: +1.579
  C: -1.579
  D: -5.010
Implied ranking: A > B > C > D
```

Nobody ever gave the model a number. Each experiment below is a one-line edit;
undo it before the next.

1. Add 100 to every learned score and recompute the loss. It is unchanged, at
   0.018. Which line of `fit_reward_model` exists purely because of that fact,
   and what would happen to the scores if you deleted it?
2. Contradict yourself. Fit on `[("A","B"), ("B","C"), ("C","A")]`. Every score
   comes back exactly 0.0 and the loss sits at 0.6931. Work out what 0.6931 is
   the logarithm of, and explain what the model is telling you.
3. Train longer. Run with `steps=` 10, 100, 500 and 2000: A's score goes
   0.897, 2.975, 5.010, 6.975 and the loss keeps falling. The scores never
   settle. What is missing from this loss that would stop them, and does the
   ranking ever change?
4. The training data is every pair among four responses, all consistent. Delete
   `("A","D")` and refit. Does A still beat D, and if so, where did that come
   from?
