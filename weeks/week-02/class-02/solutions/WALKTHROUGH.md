# W2C2 Walkthrough: Naive Bayes sentiment, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `sentiment.py` in this folder. Every code block below is
copied from it, and every printed value was produced by running it on the
shipped `TRAIN` set: 10 documents, $|V| = 40$, 32 word tokens in `pos` and 30 in
`neg`.

---

## Step 1, The prior

**The idea.** $P(c)$ is the share of the training set that belongs to class $c$,
before any words are looked at. Add-one smoothing means a class with zero
training documents still gets a small nonzero probability instead of $\log 0$.

```python
def log_prior(n_docs_in_class: int, n_docs: int, n_classes: int) -> float:
    return math.log((n_docs_in_class + 1) / (n_docs + n_classes))
```

Here the data is balanced, five documents per class, so:

```
(5 + 1) / (10 + 2) = 0.5   for both classes
```

The prior is uninformative on this corpus. Everything the classifier knows will
come out of Step 2.

**Common mistakes.** Returning the probability instead of its log (the rest of
the code adds these, so they must already be logs). Smoothing the numerator but
forgetting `+ n_classes` in the denominator, which stops the priors summing to 1
and is exactly what `test_step1_priors_sum_to_one` catches.

---

## Step 2, The likelihood

**The idea.** $P(w \mid c)$ is how much of class $c$'s text is the word $w$. The
`+1` on top pretends every vocabulary word was seen one extra time in every
class; the `+ vocab_size` on the bottom is the same `+1` counted $|V|$ times, so
the distribution still sums to 1 over the vocabulary.

```python
def log_likelihood(count_w_c: int, total_words_in_c: int, vocab_size: int) -> float:
    return math.log((count_w_c + 1) / (total_words_in_c + vocab_size))
```

`"great"` appears once in `pos` and never in `neg`:

```
pos:  (1 + 1) / (32 + 40) = 0.0278   log = -3.5835
neg:  (0 + 1) / (30 + 40) = 0.0143   log = -4.2485
```

Without the smoothing, `neg` would be $\log 0$ and a single unseen word would
veto an entire class.

**The mistake to avoid.** `total_words_in_c + 1` instead of
`total_words_in_c + vocab_size`. The result still looks like a probability and
still classifies most documents correctly, which is what makes it hard to spot.

---

## Step 3, Scoring a document

**The idea.** Bayes' rule says $P(c \mid d) \propto P(c) \prod_i P(w_i \mid c)$.
Multiplying thirty probabilities under 1 underflows to zero, so take logs and the
product becomes a sum. The denominator $P(d)$ is identical for both classes, so
it cancels and is never computed.

```python
def score(model: dict, tokens: list[str]) -> dict:
    out = {}
    for c in model["classes"]:
        s = model["log_prior"][c]
        for w in tokens:
            if w in model["vocab"]:
                s += model["log_likelihood"][c][w]
        out[c] = s
    return out
```

On `"a wonderful and moving film"`:

```
{'pos': -17.0013, 'neg': -18.9399}
```

Both are negative, because they are logs of probabilities. `pos` is the larger
(closer to zero), so `predict` returns `pos`. The gap, about 1.94, is the
model's confidence, and `predict` throws all of it away and keeps only the sign.

**Common mistakes.** Multiplying instead of adding. Starting `s` at 0 instead of
the log prior. Dropping the `in model["vocab"]` guard, which raises `KeyError` on
the first word the model has never seen.

---

## Step 4, Run it

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

Five test documents, drawn from the same tiny vocabulary as training, is not
evidence of anything. The case to discuss is
`'clever and funny but a bit slow'`: the model gets it right by counting two
positive words against one negative one, not by understanding the `but` that
flips the sentence. That is the shortcut Round 2 of the Showdown is hunting for.

---

## Given code, for reference

`count_corpus`, `train_nb`, `predict` and `prf` ship written. `prf` counts three
things over the paired lists and guards every denominator:

```python
    tp = sum(1 for g, p in zip(gold, pred) if p == target and g == target)
    fp = sum(1 for g, p in zip(gold, pred) if p == target and g != target)
    fn = sum(1 for g, p in zip(gold, pred) if p != target and g == target)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
```

Accuracy would collapse all three of those counts into one number. Precision and
recall keep "when it says pos, is it right?" separate from "does it find all the
pos?", and those two failures have very different consequences in a real system.
