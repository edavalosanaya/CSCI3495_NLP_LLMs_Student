# W7C1 Walkthrough: Decoding strategies, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `decoding.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it on the toy
logits `{"sunny": 2.0, "cloudy": 1.2, "cold": 0.6, "nice": 0.1, "banana": -3.0}`.

---

## Given, `apply_temperature`

```python
def apply_temperature(logits: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature < 1e-6:
        # Greedy: all mass on the argmax.
        best = max(logits, key=logits.get)
        return {t: (1.0 if t == best else 0.0) for t in logits}
    scaled = {t: v / temperature for t, v in logits.items()}
    m = max(scaled.values())  # for numerical stability
    exps = {t: math.exp(v - m) for t, v in scaled.items()}
    z = sum(exps.values())
    return {t: e / z for t, e in exps.items()}
```

**The `T < 1e-6` branch is a definition, not a hack.** As $T \to 0$ the softmax
converges to a one-hot on the argmax, but you cannot compute the limit by
dividing by zero. Every real inference server special-cases this the same way,
which is why "temperature 0" and "greedy decoding" are used interchangeably.

**The `- m` is the same stability trick as W5C2.** Dividing by a small `T`
*amplifies* the logits (a logit of 2.0 at `T = 0.01` becomes 200), so overflow is
far more likely here than in plain softmax. Subtracting the max is what makes low
temperatures survivable at all.

**What you should see:**

```python
>>> {k: round(v, 3) for k, v in apply_temperature(logits, 0.5).items()}
{'sunny': 0.778, 'cloudy': 0.157, 'cold': 0.047, 'nice': 0.017, 'banana': 0.0}
>>> {k: round(v, 3) for k, v in apply_temperature(logits, 1.5).items()}
{'sunny': 0.435, 'cloudy': 0.255, 'cold': 0.171, 'nice': 0.123, 'banana': 0.016}
```

**Teach these two rows against each other.** Same logits, and the ordering never
changes (temperature is monotonic, it cannot make `cloudy` beat `sunny`). What
changes is the *spread*: 0.778 vs 0.435 for the leader, and `banana` going from
effectively 0 to a real 1.6% chance of being emitted.

That is the whole creativity/coherence trade-off in two lines of numbers, and it
is worth connecting back to W1C1, where students turned this knob without knowing
what it did.

---

## Given, `greedy`

```python
def greedy(dist: dict[str, float]) -> str:
    return max(dist, key=dist.get)
```

**One line, and worth thirty seconds anyway.** Greedy is not a *bad* strategy, it
is a strategy with a specific failure mode: it is deterministic, so it produces
the same output every time, and on open-ended text it degenerates into loops
("the best way to do this is the best way to do this is..."). Holtzman et al.
2020 measured this; the repetition is not a bug in any implementation, it is what
maximizing per-token probability actually does.

For anything with one right answer (classification, extraction, structured
output) greedy is usually correct. W11's structured-output work leans on that.

---

## Step 1, `top_k_filter`

```python
def top_k_filter(dist: dict[str, float], k: int) -> dict[str, float]:
    kept = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k])
    z = sum(kept.values())
    return {t: v / z for t, v in kept.items()} if z > 0 else kept
```

**Renormalization is what makes it a distribution again.** After dropping tokens
the remaining probabilities sum to less than 1, and `sample` walks a cumulative
sum expecting a total of 1. Skipping the division makes sampling silently biased
toward the fallback branch.

**The `if z > 0` guard** handles an all-zero input, which happens if someone
top-k filters an already-filtered distribution where the survivors were rounded
to zero. Rare, but it turns a `ZeroDivisionError` into a harmless passthrough.

**What you should see:**

```python
>>> {k: round(v, 3) for k, v in top_k_filter(base, 2).items()}
{'sunny': 0.69, 'cloudy': 0.31}
```

`sunny` was about 0.55 at `T = 1.0` and is 0.69 after filtering. The mass from
the three discarded tokens was redistributed proportionally. Students sometimes
expect the kept values to be unchanged; ask them what the numbers would sum to if
so.

**The weakness to name:** `k` is fixed. If the model is certain, top-5 drags in
four tokens it had all but ruled out. If the model is genuinely torn between
twenty options, top-5 amputates sixteen of them. The threshold should depend on
the model's confidence, which is Step 4.

---

## Step 2, `top_p_filter`

```python
def top_p_filter(dist: dict[str, float], p: float) -> dict[str, float]:
    ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    kept: dict[str, float] = {}
    cum = 0.0
    for tok, prob in ordered:
        kept[tok] = prob
        cum += prob
        if cum >= p:
            break
    z = sum(kept.values())
    return {t: v / z for t, v in kept.items()} if z > 0 else kept
```

**Add first, then check.** The token is inserted into `kept` *before* the
cumulative sum is tested, which is what guarantees at least one token survives
even when `p` is smaller than the top token's probability. Writing the check
first produces an empty dict for small `p` and a crash downstream; there is a
dedicated test for exactly this.

**Note the boundary is inclusive.** The token that pushes the total over `p` is
kept, so the nucleus always covers *at least* `p` of the mass, never less. That
matches Holtzman et al.'s definition.

**What you should see:**

```python
>>> {k: round(v, 3) for k, v in top_p_filter(base, 0.8).items()}
{'sunny': 0.59, 'cloudy': 0.265, 'cold': 0.145}
```

**Put this next to Step 3's output and let the class find the difference.** Same
distribution: top-2 kept two tokens, top-p 0.8 kept three. The nucleus sized
itself to the distribution.

The demonstration that makes it click: apply top-p to a *confident* distribution
(say `{a: 0.95, b: 0.03, c: 0.02}`) and the nucleus is one token. Apply it to a
*flat* one and it is nearly all of them. Top-k gives the same count either way.
That adaptivity is why nucleus sampling is the default in most production
serving.

---

## Given, `sample`

```python
def sample(dist: dict[str, float], seed: int | None = None) -> str:
    rng = random.Random(seed)
    r = rng.random()
    cum = 0.0
    last = None
    for tok, prob in dist.items():
        last = tok
        cum += prob
        if r <= cum:
            return tok
    return last  # floating-point fallback
```

**Inverse-transform sampling**, and it is worth drawing on the board: lay the
probabilities end to end along the interval `[0, 1)`, throw a dart at a uniform
random point, and return whichever segment it lands in. Segments are as wide as
their probability, so tokens are chosen in proportion.

**`random.Random(seed)` is local**, not the global `random`, so tests reproduce
regardless of what else consumed randomness. Same discipline as W2C1's `generate`
and W4C2's sampler.

**The `return last` fallback is not dead code.** After renormalization the
probabilities sum to 1 only up to floating-point error. If they sum to
0.9999999999, a draw of `r = 0.99999999995` falls past the end of every segment
and the loop exits without returning. Returning the last token is the sane
resolution. This is a real bug in student implementations that appears roughly
one run in a billion, which is the worst kind.

**What you should see:**

```python
>>> sample(base, seed=0) == sample(base, seed=0)
True
>>> sample({"only": 1.0}, seed=3)
'only'
```

---

## Running it

```
temp=0.5 -> {'sunny': 0.778, 'cloudy': 0.157, 'cold': 0.047, 'nice': 0.017, 'banana': 0.0}
temp=1.5 -> {'sunny': 0.435, 'cloudy': 0.255, 'cold': 0.171, 'nice': 0.123, 'banana': 0.016}
greedy   -> sunny
top-2    -> {'sunny': 0.69, 'cloudy': 0.31}
top-p .8 -> {'sunny': 0.59, 'cloudy': 0.265, 'cold': 0.145}
```

**The `banana` token is the through-line.** It is in the vocabulary with logit
-3.0, deliberately absurd. Watch what each strategy does with it: temperature
0.5 buries it (0.000), temperature 1.5 gives it a genuine 1.6% chance, and both
filters remove it entirely. When a model emits something bizarre, one of these
knobs is usually why.

**Then push them to `playground.py`.** The toy distribution makes the arithmetic
visible; the real model makes the consequence audible. Students who only do the
math tend to think of temperature as an abstract parameter, and students who only
turn the knob never learn what it does. The pairing is the point of the session.

**Where this goes.** W10 asks students to hold decoding *fixed* (temperature 0,
fixed seed) while A/B testing prompts, and this is the lab that earns them the
right to that instruction: they now know exactly what varies when they do not.
