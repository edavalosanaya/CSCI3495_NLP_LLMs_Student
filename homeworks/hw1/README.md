# HW1: Text Processing & N-Gram Language Models

**Out:** Week 2, Class 1 · **Due:** Week 3, Class 2 (start of class)
**100 points** · **Weight:** 2.5% of the course grade · **Individual assignment** · **Estimated time:** 4-6 hours

---

## Learning goals
By completing this homework you will be able to:
1. Preprocess raw text into sentences and tokens with regular expressions.
2. Implement an **n-gram language model** with **add-k (Laplace) smoothing**.
3. Compute the probability of a sentence and the **perplexity** of a model on a corpus.
4. **Sample** new text from a learned distribution and reason about smoothing trade-offs.

## Background
This assignment operationalizes Week 1 (text processing) and Week 2 (n-gram language
models) and corresponds to **Jurafsky & Martin, *Speech and Language Processing* (3rd
ed.), Ch. 2-3**. An n-gram model approximates the probability of a sequence using the
**Markov assumption**, the next word depends only on the previous *n − 1* words:

> P(w₁ … w_T) ≈ ∏ₜ P(wₜ | wₜ₋ₙ₊₁ … wₜ₋₁)

Counts of unseen n-grams are zero, which makes the joint probability collapse to zero, so
we **smooth**. With **add-k smoothing** over a vocabulary `V`:

> P(w | context) = (count(context, w) + k) / (count(context) + k · |V|)

We evaluate models with **perplexity**, the inverse-probability of a held-out corpus
normalized per token: lower is better. These ideas are the conceptual ancestor of every
modern LLM, an LLM is, at heart, a very powerful next-token probability model.

You will work in **pure Python (standard library only)**: `re`, `math`, `collections`,
`random`. No NumPy, no external models.

## Files

```
hw1/
  ngram_lm.py          # <- YOU implement the TODOs here
  test_ngram_lm.py     # the tests each step below refers to
  README.md            # this handout
```

## How this homework works

This handout is a sequence of steps. Each step is one function, and **each step
ends with a test you can run**, so you always know whether you are done before
you move on. Work them in order: later steps import earlier ones.

Open a shell inside the course image, already in this homework's folder.
One command, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/homeworks/hw1 course bash
```

Everything below runs in that shell:

```bash
pytest -k step3 -q      # check ONLY step 3
pytest -q               # run every step
```

**Before you write anything, every test skips.** That is expected: the suite
detects the unfinished starter and skips rather than drowning you in failures.
The moment step 1 is implemented the tests start running for real.

**Total when you are finished: `22 passed`.**

### Step 0, Orientation (0 pts)

Nothing to write yet.

Open `ngram_lm.py` and read it top to bottom before writing anything. Note the
module constants `BOS`, `EOS` and `UNK`, and note that `NGramLM.__init__` already
stores `n` and `k` for you. Then run the suite once so you can see the starting
state:

```bash
pytest -q
```

You should get `22 skipped`. That is the suite telling you it found an unfinished
starter, not a broken install.

### Step 1, `tokenize` (8 pts)

**Write** `tokenize(text)`. Lowercase the text, then split it into tokens, where a token is either a run of word characters or a single punctuation character. One regex does the whole job: `re.findall(r"\w+|[^\w\s]", text.lower())`.

**Done when** `pytest -k step1 -q` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> from ngram_lm import tokenize
>>> tokenize("Hello, world!")
['hello', ',', 'world', '!']
```

**Why it matters.** Every later step counts tokens, so a tokenizer that silently drops punctuation or keeps capitalization changes every probability you compute downstream. This is the step where 'the' and 'The' become the same word, or don't.

### Step 2, `sentences` (7 pts)

**Write** `sentences(text)`. Split on sentence-final punctuation (`.!?`, with a run like `!?` counting as one break), tokenize each piece, and drop pieces that come back empty.

**Done when** `pytest -k step2 -q` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> from ngram_lm import sentences
>>> sentences("A cat sat. It ran!")
[['a', 'cat', 'sat'], ['it', 'ran']]
```

**Why it matters.** An n-gram model is trained per sentence, not across the whole document. Without this split the model learns that the last word of one sentence predicts the first word of the next, which is nonsense it will happily generate later.

### Step 3, `pad` and `ngrams` (10 pts)

**Write** `pad(tokens)`, which wraps a sentence in `n - 1` copies of `<s>` and a single `</s>`, and `ngrams(tokens)`, which slides a window of length `n` over an already-padded list. If there are fewer tokens than the window, return `[]` rather than crashing.

**Done when** `pytest -k step3 -q` prints `5 passed, 17 deselected`.

**Check it by hand**

```python
>>> from ngram_lm import NGramLM
>>> m = NGramLM(n=2)
>>> m.pad(["a", "b"])
['<s>', 'a', 'b', '</s>']
>>> m.ngrams(m.pad(["a", "b"]))
[('<s>', 'a'), ('a', 'b'), ('b', '</s>')]
```

**Why it matters.** The padding is what lets the model assign a probability to the FIRST word of a sentence, which otherwise has no context, and `</s>` is what lets it decide to stop. Without `</s>` your generator in step 8 never terminates on its own.

### Step 4, `fit` and `_map` (15 pts)

**Write** `fit(corpus)`, which builds `self.vocab` and the n-gram and context counts, and `_map(token)`, which returns the token if it is in the vocabulary and `UNK` otherwise. The vocabulary contains every training word plus `</s>` and `<unk>`, but **not** `<s>`: you never predict a sentence-start.

**Done when** `pytest -k step4 -q` prints `3 passed, 19 deselected`.

**Check it by hand**

```python
>>> CORPUS = [["i", "am", "sam"],
...           ["sam", "i", "am"],
...           ["i", "do", "not", "like", "green", "eggs"]]
>>> m = NGramLM(n=2, k=1.0).fit(CORPUS)
>>> len(m.vocab)
10
>>> m._map("sam"), m._map("aardvark")
('sam', '<unk>')
```

**Why it matters.** `|V|` appears in the denominator of every smoothed probability, so an off-by-one here (counting `<s>`, forgetting `<unk>`) shifts every number in step 5. It is the single most common source of 'my probabilities almost sum to 1'.

### Step 5, `prob` (20 pts)

**Write** `prob(token, context)`, the add-k smoothed conditional probability

> P(w | context) = (count(context, w) + k) / (count(context) + k * |V|)

Map both the token and every element of the context through `_map` first, so an unseen word is scored as `<unk>` instead of crashing.

**Done when** `pytest -k step5 -q` prints `4 passed, 18 deselected`.

**Check it by hand**

```python
>>> CORPUS = [["i", "am", "sam"],
...           ["sam", "i", "am"],
...           ["i", "do", "not", "like", "green", "eggs"]]
>>> m = NGramLM(n=2, k=1.0).fit(CORPUS)
>>> round(m.prob("am", ("i",)), 4)      # count("i am")=2, count("i")=3, |V|=10
0.2308
>>> round(m.prob("zzz", ("i",)), 4)     # unseen -> <unk>, never zero
0.0769
>>> round(sum(m.prob(w, ("i",)) for w in m.vocab), 6)
1.0
```

**Why it matters.** Check that last line every time you touch this function. A distribution that does not sum to 1 is not a distribution, and perplexity computed from it is meaningless even though it will still print a number.

### Step 6, `sentence_logprob` (8 pts)

**Write** `sentence_logprob(tokens)`: pad the sentence, walk its n-grams, and sum `log(prob(...))`. Work in log space, not by multiplying probabilities.

**Done when** `pytest -k step6 -q` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> CORPUS = [["i", "am", "sam"],
...           ["sam", "i", "am"],
...           ["i", "do", "not", "like", "green", "eggs"]]
>>> m = NGramLM(n=2, k=1.0).fit(CORPUS)
>>> round(m.sentence_logprob(["i", "am", "sam"]), 4)
-6.5162
```

**Why it matters.** Multiplying twenty probabilities of 0.05 underflows to exactly 0.0 in float64, and then the perplexity is `inf`. Summing logs is not a stylistic preference; it is the only version that works on a real corpus.

### Step 7, `perplexity` (7 pts)

**Write** `perplexity(corpus)`: accumulate the total log-probability and the total number of **predicted** tokens across every sentence, then return `exp(-total_logprob / total_tokens)`. Each sentence contributes its own `</s>` to the token count.

**Done when** `pytest -k step7 -q` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> CORPUS = [["i", "am", "sam"],
...           ["sam", "i", "am"],
...           ["i", "do", "not", "like", "green", "eggs"]]
>>> m = NGramLM(n=2, k=1.0).fit(CORPUS)
>>> round(m.perplexity(CORPUS), 4)
5.4013
```

**Why it matters.** Perplexity is the number you will compare models with for the rest of the course. Read it as the model's average branching factor: 5.4 means that at each token it is about as uncertain as if it were choosing uniformly among five words.

### Step 8, `generate` (10 pts)

**Write** `generate(max_len, seed)`. Start from the padded context, sample the next token from `prob` over the whole vocabulary using `random.Random(seed)`, append it, slide the context, and stop at `</s>` or `max_len`. Do not emit `<s>` or `</s>` in the returned list.

**Done when** `pytest -k step8 -q` prints `2 passed, 20 deselected`.

**Check it by hand**

```python
>>> CORPUS = [["i", "am", "sam"],
...           ["sam", "i", "am"],
...           ["i", "do", "not", "like", "green", "eggs"]]
>>> m = NGramLM(n=2, k=1.0).fit(CORPUS)
>>> m.generate(max_len=8, seed=0)
['not', 'like', 'eggs', '<unk>', 'green', 'eggs', 'like', 'do']
>>> m.generate(max_len=8, seed=0) == m.generate(max_len=8, seed=0)
True
```

**Why it matters.** Passing the seed through is what makes your homework gradeable and your bugs reproducible. Note also that `<unk>` can be sampled: smoothing gave it probability mass, so the model can generate a word it never saw. That trade-off is the point of the reflection.

### Step 9, Run the whole thing (0 pts)

```bash
pytest -q
```

Every step green means `22 passed`. If a step you finished earlier has gone red,
you broke it with a later change; fix that before you submit.

## Written reflection (15 pts)

Answer in `ngram_lm.py`'s module docstring or a short `REFLECTION.md`, a
paragraph each:

1. You set `k = 1.0`. What happens to your perplexity as `k` grows, and why? Run it
   at `k = 0.01`, `k = 1`, and `k = 10` on the same corpus and report the numbers.
2. Step 8 can generate `<unk>`. Explain in your own words why smoothing makes that
   possible, and say whether you think it is a bug or a feature.
3. A trigram model has more context than a bigram model. Give one concrete reason it
   can still be *worse* on held-out text.

## What to submit

- `ngram_lm.py` with every TODO filled in and `pytest -q` fully green.
- Your reflection (in the module docstring or `REFLECTION.md`).
- The `AI-USE:` note described below.

Partial credit follows the tests: each step is worth the points listed above, and a
step whose tests pass earns them. Code that does not import earns at most the
reflection points, so submit something that runs even if it is incomplete.

## AI-use disclosure (required)

Per the syllabus, you may use LLM tools as coding assistants, but you must
**disclose** it (which tool, for what), be able to **explain every line** you
submit, and write the reflection in your own words. Put a short `AI-USE:` note
in your file header. Undisclosed AI use is an academic-integrity violation.
