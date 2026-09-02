# W2C2 Lab: Sentiment Showdown

## 1. Learning objective

Implement the three equations behind Naive Bayes, use them to classify movie
reviews, and measure the result with precision, recall and F1.

You write three short functions in `sentiment.py`. The counting, the plumbing
and the reporting are already written for you.

## 2. Understanding the math

![Training Naive Bayes: the prior and the add-one-smoothed likelihood are just counts](../lecture/visuals/training-nb.png)

Let $N_c$ be the documents of class $c$ out of $N$ total, $|C|$ the number of
classes, $T_c$ the word tokens in class $c$, and $|V|$ the vocabulary size. Both
estimates are add-one smoothed, so nothing is ever $\log 0$:

$$P(c) = \frac{N_c + 1}{N + |C|} \qquad P(w \mid c) = \frac{\text{count}(w, c) + 1}{T_c + |V|}$$

A document is scored in log space, because multiplying many small probabilities
underflows. Its label is the higher-scoring class:

$$\hat{c} = \arg\max_{c} \Big[ \log P(c) + \sum_{i} \log P(w_i \mid c) \Big]$$

![Precision, recall, and F1 as formulas over TP, FP, FN](../lecture/visuals/precision-recall.png)

$$P = \frac{TP}{TP+FP} \qquad R = \frac{TP}{TP+FN} \qquad F_1 = \frac{2PR}{P+R}$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-02/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `log_prior`

Return $\log P(c)$ from the counts. One line.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 5. Implement `log_likelihood`

Return $\log P(w \mid c)$ from the counts. One line, and smoothing changes
both halves of the fraction, not just the top.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 6. Implement `score`

Return a score per class: the class's log prior, plus the log likelihood of
every token the model has seen before. Tokens outside the vocabulary are
ignored. The docstring says which key of `model` holds each of those, and the
sum in section 2 says how they combine. Add the logs; do not multiply.

```bash
pytest -k step3 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 7. Run it, then break it

```bash
python sentiment.py
```

```
============================================================
Sentiment Showdown, Naive Bayes from scratch
============================================================
  [OK ] gold=pos  pred=pos   'a great and moving story i loved it'
  [OK ] gold=pos  pred=pos   'brilliant and beautifully shot'
  [OK ] gold=neg  pred=neg   'a dull and boring waste of time'
  [OK ] gold=neg  pred=neg   'the worst and most forgettable mess'
  [OK ] gold=pos  pred=pos   'clever and funny but a bit slow'

  pos-class  precision=1.00  recall=1.00  F1=1.00
```

Five test documents drawn from the training vocabulary. That 1.00 is not
evidence of anything, so spend the rest of the period finding out what it hides.
Experiments 1 and 2 are one-line edits, so undo each before the next.
Experiment 4 builds on 3, so keep 3 in place.

1. Drop the smoothing. In `log_likelihood`, change `+ 1` to `+ 0` and re-run.
   What is the error, and what does it say about a word the class never used?
2. Mislabel one review. Change the label of `TRAIN[0]` from `"pos"` to `"neg"`
   and re-run. Precision stays at 1.00 but recall falls to 0.67. Why did only
   one of the two move?
3. Try negation. Add `("not funny not clever not good", "neg")` to `TEST` and
   re-run. The model calls it `pos` and precision drops to 0.75. What would it
   have to represent to get this right, that a bag of words cannot?
4. Fix it with data. Keeping that review in `TEST`, add negative training
   snippets containing "not" to `TRAIN` until the model gets it right. One is
   enough to return to 1.00. Then make your added snippets reuse positive words,
   like `("not funny at all", "neg")`. Recall drops to 0.67 and
   `'clever and funny but a bit slow'` flips to `neg`. Why did teaching the
   model about "not" cost it "funny"?
