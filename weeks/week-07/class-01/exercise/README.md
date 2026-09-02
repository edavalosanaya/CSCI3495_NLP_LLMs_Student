# W7C1 Lab: Decoding Strategies

## 1. Learning objective

Control what a language model says by reshaping its probability distribution
before sampling: truncate it to the top k tokens, or to the smallest set that
covers probability p.

You write two functions in `decoding.py`. Temperature, greedy and the sampler
are given.

## 2. Understanding the math

![Decoding strategies compared: greedy, top-k, top-p](../lecture/visuals/decoding-strategies.png)

Temperature rescales the logits before the softmax. Below 1 sharpens the
distribution, above 1 flattens it:

$$P(w) = \frac{\exp(z_w / T)}{\sum_v \exp(z_v / T)}$$

Top-k keeps a fixed number of tokens. Top-p keeps however many are needed to
reach probability $p$, so it adapts: a confident distribution keeps one token,
an uncertain one keeps many. Both must renormalize what survives, because a
truncated distribution no longer sums to 1.

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-07/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `top_k_filter`

Keep the k most likely tokens, then divide by what is left so the result sums
to 1 again.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 8 deselected
```

## 5. Implement `top_p_filter`

Keep tokens in order until the running total reaches p, including the one that
crosses it, then renormalize. Never return an empty distribution.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 7 deselected
```

## 6. Run it, then break it

```bash
python decoding.py
```

```
temp=0.5 -> {'sunny': 0.778, 'cloudy': 0.157, 'cold': 0.047, 'nice': 0.017, 'banana': 0.0}
temp=1.5 -> {'sunny': 0.435, 'cloudy': 0.255, 'cold': 0.171, 'nice': 0.123, 'banana': 0.016}
greedy   -> sunny
top-2    -> {'sunny': 0.69, 'cloudy': 0.31}
top-p .8 -> {'sunny': 0.59, 'cloudy': 0.265, 'cold': 0.145}
```

`banana` is the nonsense token, and its fate across those lines is the whole
lesson.

1. Feed both filters a flat distribution, `{'a': .25, 'b': .25, 'c': .25,
   'd': .25}`, then a peaked one, `{'a': .97, 'b': .01, 'c': .01, 'd': .01}`.
   Top-k keeps 2 tokens in both cases. Top-p at 0.8 keeps 4 and then 1. Which
   behaviour do you want from a model that is sometimes sure and sometimes not?
2. Push top-p to its edges. `top_p_filter(base, 0.0)` returns `{'sunny': 1.0}`
   and `top_p_filter(base, 1.0)` returns everything. Why does p of 0 return one
   token instead of none, and which line of your code decides that?
3. Give top-k an impossible k. Call `top_k_filter(base, 99)` on a five-token
   distribution. Nothing is dropped and the probabilities are unchanged. Should
   that be an error instead? Argue either way.
4. Follow `banana`. It holds 0.004 of the mass at T=1.0, rises to 0.0155 at
   T=1.5, and rounds to 0.0 at T=0.5. Top-p at 0.8 cuts it, top-k at 2 cuts it.
   Name a temperature and a p that would let it through, and say what that
   would do to the generated text.
