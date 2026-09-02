# W7C2 Lab: Scaling Laws

## 1. Learning objective

Build the measuring instrument behind every scaling claim: score a model's
answers, then check whether accuracy actually holds up as models get bigger.

You write two functions in `scaling.py`. Answer normalization and the lenient
match are given, and `measure.py` runs the same suite against real Ollama
models.

## 2. Understanding the math

![Kaplan et al. 2020, Fig. 1: loss falls as a power law in compute, data, and parameters](../lecture/visuals/assets/kaplan-2020-fig-1.png)

Training compute is roughly six FLOPs per parameter per token, and Chinchilla's
rule of thumb says a compute budget is best spent at about twenty tokens per
parameter, not on parameters alone:

$$C \approx 6\,N D \qquad \text{(training compute budget)} \qquad D_{\text{opt}} \approx 20\,N \qquad \text{(Chinchilla rule of thumb)}$$

![Hoffmann et al. 2022 (Chinchilla), Fig. 3: for a fixed FLOP budget, too big is as wasteful as too small](../lecture/visuals/assets/chinchilla-2022-fig-3.png)

What you measure here is the right-hand side: accuracy over $n$ items, and
whether it is non-decreasing across models ordered smallest to largest:

$$\text{accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\big[\text{is\_correct}(o_i, t_i)\big] \qquad \text{scaling\_trend} = \text{True} \iff a_1 \le a_2 \le \dots \le a_m$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-07/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `accuracy`

Count the items `is_correct` accepts and divide. Handle the empty run.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 5. Implement `scaling_trend`

One line: is the accuracy list non-decreasing in the order given?

```bash
pytest -k step2 -q
```

```
...                                                                      [100%]
3 passed, 5 deselected
```

## 6. Run it, then question it

```bash
python scaling.py
```

```
small-model accuracy: 0.60
large-model accuracy: 1.00
accuracy non-decreasing with size? True
```

Those two "models" are hard-coded answer lists, not models. Before trusting the
verdict, attack the instrument.

1. Look at what the lenient match forgives. The large model answered `"Paris."`
   and `"7 days"` and both scored correct, because the target is a substring.
   Now imagine an answer of `"not Paris"`. Would `is_correct` accept it? Try
   it, and decide whether that is a bug or an acceptable cost.
2. Test the trend function's edges. `scaling_trend({})` is `True` and
   `scaling_trend({"a": 0.3})` is `True`. Is "no evidence" the same answer as
   "evidence that scaling helps"? Say what you would rather these returned.
3. Break the ordering promise. `scaling_trend` never sorts; it trusts the
   caller to pass models smallest-first. Pass them largest-first instead and
   watch a real improvement report as `False`. Where should that ordering be
   enforced?
4. Run it on real models: `python measure.py`. Compare the accuracies you get
   from `qwen2.5:0.5b` and `qwen2.5:1.5b` against the simulated 0.60 and 1.00.
   Five questions is a very small suite. How many would you want before
   claiming a scaling law?
