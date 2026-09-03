# W9C2 Lab: Evaluation Harness

## 1. Learning objective

Build the measuring instruments for LLM output: flag hallucinations on
unanswerable questions, and catch an LLM judge that decides by position rather
than by content.

You write three functions in `eval_harness.py`. Normalization, exact match,
containment and accuracy are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-09/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `is_hallucination`

![Metrics compared on one worked confusion matrix](../lecture/visuals/metrics.png)

The two string metrics differ in how much they forgive. Exact match demands the
normalized strings be equal; containment only asks that the gold answer appear
somewhere in the prediction:

$$\text{EM}(p, g) = \mathbf{1}\big[\mathrm{norm}(p) = \mathrm{norm}(g)\big] \qquad \text{Contains}(p, g) = \mathbf{1}\big[\mathrm{norm}(g) \subseteq \mathrm{norm}(p)\big]$$

Flag only unanswerable items where the model failed to abstain. An answerable
item is never flagged, however wrong the answer is.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 11 deselected
```

## 4. Implement `judge_pairwise`

Ask the judge twice with the answers swapped, translate each verdict from slot
to answer, and report whether the two agree.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 11 deselected
```

## 5. Implement `position_bias_rate`

![Judge biases: the same pair, judged in both orders, flips the verdict](../lecture/visuals/judge-biases.png)

A judge's verdict is only evidence if it survives swapping the two answers:

$$\text{consistent} \iff \mathrm{judge}(q, a_1, a_2) \text{ and } \mathrm{judge}(q, a_2, a_1) \text{ name the same winner}$$

The fraction of pairs that came back inconsistent.

```bash
pytest -k step3 -q
```

```
..                                                                       [100%]
2 passed, 11 deselected
```

## 6. Run it, then question it

```bash
python eval_harness.py
```

```
== Deterministic biased judge (always picks the first slot) ==
Q: Explain why the sky is blue.
  run1 winner: ans1  run2 winner: ans2  consistent: False
Position-bias (inconsistency) rate: 100%

== Live judge: qwen2.5:0.5b (with swap check) ==
Q: Explain why the sky is blue.
  run1 winner: ans1  run2 winner: ans2  consistent: False
Position-bias (inconsistency) rate: 100%
```

The deliberately broken judge and the real model score the same: 100%.

1. Score a judge that always says "tie". Pass `lambda q, a, b: "tie"` to
   `position_bias_rate`: it scores 0.0, the same as a perfectly fair judge.
   Consistency is clearly not sufficient. What else would you have to measure?
2. Compare against a content-aware judge. Write one that prefers whichever
   answer contains the word `scatter` and score it: 0.0, against the biased
   judge's 1.0. Look at `judge_pairwise` for a single pair in each case and say
   exactly which field differs.
3. Skip the slot-to-answer translation. Compare the two RAW verdicts instead of
   calling `_winner_of`. The position-biased judge now looks perfectly
   consistent, because it said "A" both times. Explain the bug in one sentence.
4. The live judge is `qwen2.5:0.5b`. Before concluding that LLM judges are
   worthless, name two things about this setup other than model quality that
   could produce a 100% flip rate, and how you would rule each out.
