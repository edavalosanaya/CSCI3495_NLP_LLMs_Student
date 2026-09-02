# W2C1 Walkthrough: N-gram Bard, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on**; the whole point of the lab is doing the next step yourself.

The complete file is `ngram_lm.py` in this folder. Everything below is taken
from it, and every printed value was produced by actually running it.

Throughout, `TRAIN = ["the cat sat", "the cat ran", "a cat sat"]` (the tests'
corpus), which has vocabulary `{the, cat, sat, ran, a, </s>}`, so $V = 6$.

---

## Given, `count_ngrams`

**The idea.** A padded sentence is just a list of tokens. Slide a window of
width `n` across it. Each window is an n-gram; drop its last word and you have
the context that predicted it. Count both, because every probability later is
one divided by the other.

```python
def count_ngrams(sentences: list[str], n: int) -> dict:
    ngram = defaultdict(int)
    context = defaultdict(int)
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            ngram[gram] += 1
            context[gram[:-1]] += 1
    return {
        "n": n,
        "vocab": vocab(sentences),
        "ngram": dict(ngram),
        "context": dict(context),
    }
```

**Line by line.**

- `pad(tokenize(s), n)` turns `"the cat sat"` into `['<s>', 'the', 'cat', 'sat', '</s>']`
  for `n=2`. For `n=3` you get two `<s>`, because a trigram needs two words of
  context before the first real word.
- `range(len(toks) - n + 1)` is every valid window start. For 5 tokens and
  `n=2` that is `0..3`, giving the four bigrams.
- `gram[:-1]` is "everything but the last word", which is exactly the context.
  For a unigram (`n=1`) it is the empty tuple `()`, and that is correct: a
  unigram model has no context, so all counts pile into one bucket.
- `dict(...)` on the way out just stops a `defaultdict` from silently inventing
  zero entries when someone looks up a missing key later.

**What you should see:**

```python
>>> m = count_ngrams(TRAIN, 2)
>>> m["ngram"][("the", "cat")]
2
>>> m["context"][("the",)]
2
>>> m["ngram"][("sat", "</s>")]
2
>>> len(m["vocab"])
6
```

**Common mistakes.**

- Forgetting to pad, so the model never learns which words start or end a
  sentence, and `generate` has nothing to begin from.
- Counting contexts in a separate pass over the text. You do not need to: every
  n-gram contributes exactly one context, so count them together and they can
  never disagree.
- Putting `<s>` in the vocabulary. It is never *predicted*, only conditioned on,
  which is why `vocab()` seeds itself with `{EOS}` and not `{BOS, EOS}`.

---

## Step 1, `prob`

**The idea.** The raw estimate is `count(context, word) / count(context)`. The
problem is that a corpus this small has never seen most word pairs, so most
probabilities would be exactly 0, and a single unseen pair would make the whole
sentence probability 0 (and perplexity infinite). Add-one smoothing pretends
every possible n-gram was seen one extra time.

```python
def prob(model: dict, context: tuple, word: str) -> float:
    v = len(model["vocab"])
    c_ctx = model["context"].get(context, 0)
    c_gram = model["ngram"].get(context + (word,), 0)
    return (c_gram + 1) / (c_ctx + v)
```

**Why `+ V` in the denominator, not `+ 1`?** Because you added one imaginary
count to *every* word that could follow this context, and there are $V$ of them.
The denominator has to absorb all $V$ of those invented counts or the
distribution stops summing to 1. That single detail is what the second test
checks, and it is the one students most often get wrong.

**What you should see** (with $V = 6$ and `count(the) = 2`):

```python
>>> m = count_ngrams(TRAIN, 2)
>>> prob(m, ("the",), "cat")        # (2 + 1) / (2 + 6)
0.375
>>> prob(m, ("the",), "ran")        # unseen: (0 + 1) / (2 + 6)
0.125
>>> sum(prob(m, ("the",), w) for w in m["vocab"])
1.0
```

**Common mistakes.**

- Indexing with `model["ngram"][key]` instead of `.get(key, 0)`, which raises
  `KeyError` on the first unseen pair.
- Building the lookup key as `(context, word)` (a 2-tuple containing a tuple)
  rather than `context + (word,)` (a flat tuple), so nothing ever matches.

---

## Given, `generate`

**The idea.** Generation is a loop: look at the current context, score every
word that could come next, sample one, slide the context forward, repeat until
the model produces `</s>`.

```python
def generate(model: dict, n: int, max_len: int = 20, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    words = sorted(model["vocab"])  # sorted for determinism
    context = (BOS,) * (n - 1)
    out: list[str] = []
    for _ in range(max_len):
        weights = [prob(model, context, w) for w in words]
        nxt = rng.choices(words, weights=weights, k=1)[0]
        if nxt == EOS:
            break
        out.append(nxt)
        if n > 1:
            context = (context + (nxt,))[1:]
    return out
```

**The two determinism traps**, both of which the test catches:

1. `random.Random(seed)` creates a **local** generator. Calling the global
   `random.choices` instead would make your output depend on whatever else in
   the process consumed randomness first.
2. `sorted(model["vocab"])` fixes the iteration order. Python set order varies
   between runs, so without the `sorted` the same seed picks a different word
   even though the RNG behaved identically.

**The context slide.** `(context + (nxt,))[1:]` appends the new word and drops
the oldest, keeping the context exactly `n-1` long. For a bigram that means the
context is simply the word you just emitted. For a unigram, `n > 1` is false and
the context stays `()` forever, which is right: a unigram model ignores history.

**What you should see:**

```python
>>> m = count_ngrams(TRAIN, 2)
>>> generate(m, 2, seed=3)
['a', 'cat', 'ran', 'ran', 'ran']
```

Different word order in your vocabulary can give a different sentence. What must
hold is that the same seed reproduces its own output, and that no `<s>` or
`</s>` leaks into the returned list.

Note the `ran ran ran`: with add-one smoothing on a three-sentence corpus, every
word has a real chance of following every other word. Fluency comes from data,
not from the algorithm.

---

## Step 2, `perplexity`

**The idea.** Perplexity asks: on held-out text, how surprised was the model, on
average? Compute the probability the model assigns to each token, average the
log-probabilities, and exponentiate back.

```python
def perplexity(model: dict, n: int, sentences: list[str]) -> float:
    log_sum = 0.0
    count = 0
    for context, word in iter_predictions(n, sentences):
        log_sum += math.log(prob(model, context, word))
        count += 1
    if count == 0:
        return float("inf")
    return math.exp(-log_sum / count)
```

**Why the loop starts at `n - 1`.** The first `n-1` tokens are the `<s>` padding.
They are context, never predictions, so scoring them would be scoring the model
on something it was handed for free. The loop *does* run through the final
`</s>`, because deciding to stop is a real prediction.

**Why logs.** Each `prob` is well under 1, so multiplying a few hundred of them
underflows to `0.0` and `exp` of that is meaningless. Summing logs keeps the
numbers in a sane range; the negative sign and the `exp` at the end just undo
the log so the result reads as "effective number of choices per word".

**What you should see:**

```python
>>> uni = count_ngrams(TRAIN, 1)
>>> bi  = count_ngrams(TRAIN, 2)
>>> round(perplexity(uni, 1, ["the cat sat"]), 4)
5.1962
>>> round(perplexity(bi, 2, ["the cat sat"]), 4)
2.8284
```

The bigram is about half as perplexed, because knowing the previous word
genuinely narrows down what comes next.

**Common mistakes.**

- Starting the loop at `0` and scoring the `<s>` padding, which inflates or
  deflates the score depending on `n` and makes models with different `n`
  incomparable.
- Dividing by the number of *sentences* instead of the number of predicted
  tokens.
- Returning `-log_sum / count` (the average log-probability, sometimes called
  cross-entropy) without the `exp`. It is a fine quantity, it just is not
  perplexity.

---

## Running it

```
============================================================
N-gram Bard, train, babble, and score with perplexity
============================================================

[unigram]  perplexity on held-out =   15.39
   babble: eggs am not

[bigram]  perplexity on held-out =    8.64
   babble: ham am mat

[trigram]  perplexity on held-out =   10.63
   babble: ham and mat a in green a in
```

**The teaching moment.** Fluency rises with `n`: the trigram's output has real
phrase structure, the unigram's is word salad. But perplexity does **not** keep
falling, the trigram (10.63) scores worse than the bigram (8.64).

That gap is overfitting, and it is worth spending class time on. With ten short
sentences, almost every trigram context occurs exactly once, so the model
becomes confident about continuations it has one shred of evidence for. When the
held-out sentence presents a context it never saw, add-one smoothing spreads the
probability thinly over the whole vocabulary and the model gets punished. The
bigram, with fewer and better-attested contexts, generalizes better.

This is the same bias/variance story that runs through the rest of the course.
More capacity fits the training data better and the held-out data worse, until
you give it more data.
