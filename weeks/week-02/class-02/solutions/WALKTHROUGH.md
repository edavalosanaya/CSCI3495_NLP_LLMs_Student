# W2C2 Walkthrough: Naive Bayes sentiment, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `sentiment.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it on the
shipped `TRAIN` set (10 documents, `|V| = 40`).

---

## Step 1, Train

**The idea.** Naive Bayes "training" is one counting pass. You need, per class:
the number of documents, the number of word tokens, and a count for each word.
Everything else is arithmetic on those counts.

```python
    vocab: set[str] = set()
    for doc, y in zip(docs, labels):
        toks = tokenize(doc)
        word_counts[y].update(toks)
        vocab.update(toks)

    v = len(vocab)
    class_total = {c: sum(word_counts[c].values()) for c in CLASSES}
    # Add-one smoothing on the prior too, so a class with no training docs
    # gets a tiny nonzero probability instead of log(0).
    log_prior = {
        c: math.log((class_docs[c] + 1) / (n + len(CLASSES))) for c in CLASSES
    }
    log_likelihood = {c: {} for c in CLASSES}
    for c in CLASSES:
        denom = class_total[c] + v
        for w in vocab:
            log_likelihood[c][w] = math.log((word_counts[c][w] + 1) / denom)
```

**The denominator is the part to slow down on.** `class_total[c] + v`, not
`class_total[c] + 1`. You added one imaginary occurrence of *every* vocabulary
word to this class, so the total must absorb all `v` of them. Get this wrong and
the per-class likelihoods stop summing to 1; the classifier often still works,
which makes the bug hard to notice, and that is exactly why the exercise checks
it.

**Note the likelihood table is dense.** Every class stores an entry for every
vocabulary word, including words it never saw (they get count 0, hence
probability `1 / denom`). With `|V| = 40` that is cheap and keeps `score` simple.
A real implementation would store only observed words and compute the fallback on
demand, which is why `score` handles the missing-key case anyway.

**Why smooth the prior.** With 5 and 5 documents it changes 0.5 to 0.5 and looks
pointless. It matters the moment a student adds their own data in Round 1 of the
Showdown and creates a class with very few documents, or when an empty class
would otherwise produce `log(0)` and crash.

**What you should see:**

```python
>>> m = train_nb([d for d, _ in TRAIN], [y for _, y in TRAIN])
>>> {c: round(math.exp(m["log_prior"][c]), 4) for c in m["classes"]}
{'pos': 0.5, 'neg': 0.5}
>>> len(m["vocab"])
40
>>> m["class_total"]
{'pos': 32, 'neg': 30}
```

---

## Step 2, Score

**The idea.** Sum the log prior and the log likelihood of every token. Log space
turns a product of forty small probabilities into a sum, which does not
underflow.

```python
def score(model: dict, tokens: list[str]) -> dict:
    v = len(model["vocab"])
    out = {}
    for c in model["classes"]:
        s = model["log_prior"][c]
        fallback = math.log(1 / (model["class_total"][c] + v))
        for w in tokens:
            if w in model["vocab"]:
                s += model["log_likelihood"][c].get(w, fallback)
        out[c] = s
    return out
```

**Two different kinds of "unseen", and they are handled differently.**

- A word **not in `model["vocab"]`** at all (it never appeared in any training
  document) is **skipped**. The model has no evidence about it either way, and
  scoring it would just add the same constant to both classes.
- A word **in the vocabulary but absent from class `c`** gets the `fallback`,
  `1 / (class_total[c] + v)`, which is exactly the add-one probability for a
  count of zero. Skipping this case instead is the classic bug: it deletes the
  evidence that a strongly negative word is missing from the positive class, and
  the classifier becomes overconfident.

In this dense implementation the `.get(w, fallback)` never actually fires,
because Step 1 stored every vocabulary word for every class. It is there so the
function stays correct if someone rewrites `train_nb` to store only observed
words, which is a reasonable stretch goal.

**What you should see:**

```python
>>> t = tokenize("a wonderful and moving film")
>>> {c: round(v, 4) for c, v in score(m, t).items()}
{'pos': -17.0013, 'neg': -18.9399}
```

Both are negative because they are logs of probabilities. The 1.94 gap is the
model's margin, and it is worth pointing out that the number itself is not a
probability of anything; only the comparison between classes is meaningful.

---

## Step 3, Predict

```python
def predict(model: dict, tokens: list[str]) -> str:
    scores = score(model, tokens)
    return max(scores, key=scores.get)
```

**The missing denominator.** Bayes' rule says
$P(c \mid d) = P(c)P(d \mid c) / P(d)$, and we never compute $P(d)$. We do not
have to: it is identical for both classes, so it cannot change which one is
larger. This is why the returned numbers are log-scores rather than log
probabilities, and why they do not exponentiate to anything that sums to 1.

If you *do* want calibrated probabilities (for a threshold knob, say), normalize
the exponentiated scores across classes. That is the softmax that shows up again
in Week 4.

**What you should see:**

```python
>>> predict(m, tokenize("a wonderful and moving film"))
'pos'
```

---

## Step 4, Evaluate

**The idea.** Three counts over the paired lists, then three formulas. The only
subtlety is refusing to divide by zero.

```python
def prf(gold: list[str], pred: list[str], target: str = "pos") -> dict:
    tp = sum(1 for g, p in zip(gold, pred) if p == target and g == target)
    fp = sum(1 for g, p in zip(gold, pred) if p == target and g != target)
    fn = sum(1 for g, p in zip(gold, pred) if p != target and g == target)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}
```

**Read the three comprehensions as questions about the prediction.** TP and FP
both start "I predicted target"; they differ on whether gold agreed. FN starts "I
did not predict target" but gold says I should have. True negatives never appear,
which is the whole point: precision and recall deliberately ignore the correct
rejections, so a class that is rare cannot be masked by a large easy majority.

**The zero-denominator guards are not defensive padding.** A model that never
predicts `pos` has `tp + fp == 0`, and "precision" is genuinely undefined, not
zero. Reporting 0.0 is a convention, and it is the convention scikit-learn uses
too (with a warning). Say this out loud in class, because students often assume a
0.0 means the model tried and failed rather than never tried.

**What you should see:**

```python
>>> prf(["pos", "neg"], ["pos", "neg"])
{'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
>>> prf(["pos", "pos", "neg"], ["pos", "neg", "pos"])
{'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
```

The second case: TP = 1 (first item), FP = 1 (third item, predicted pos, gold
neg), FN = 1 (second item, gold pos, predicted neg). So P = 1/2, R = 1/2, and F1
= 1/2.

---

## Step 5, Run the whole thing

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

**Do not let a perfect score pass without comment.** Five test documents drawn
from the same 40-word vocabulary as training is not evidence of generalization,
and a class that walks away thinking Naive Bayes "solved" sentiment has learned
the wrong lesson. Two things to make explicit:

1. **The test set is too small and too similar.** The honest read is "nothing
   broke", not "the model is good". This is a live preview of the evaluation
   discipline in Week 9 and of the contamination problem in W9C2.
2. **The last case is right for the wrong reason.** "clever and funny but a bit
   slow" scores pos because two positive words outweigh one negative one. The
   model has no representation of "but", which in English usually signals that
   the clause *after* it carries the writer's real verdict. Flip the sentence to
   "a bit slow but clever and funny" and bag-of-words returns exactly the same
   answer, because word order is not part of the model at all.

That second point is the bridge to the rest of the course. Round 2 of the
Showdown (stump the model) is where students discover it themselves, and
sarcasm, negation scope, and mixed sentiment are the three reliable ways in.
