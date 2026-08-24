# HW1: Text Processing & N-Gram Language Models

**Out:** Week 2, Class 1 · **Due:** Week 3, Class 2 (start of class)
**Weight:** 5% of course grade · **Individual assignment** · **Estimated time:** 4-6 hours

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
  test_ngram_lm.py     # public tests (run these as you go)
  README.md            # this handout
```

## Tasks

### Task 1: Preprocessing (15 pts)
Implement in `ngram_lm.py`:
1. **`tokenize(text)`**, lowercase, then split into tokens where a token is a run of
   word characters or a single punctuation character. *(Hint: `re.findall(r"\w+|[^\w\s]", text.lower())`.)*
2. **`sentences(text)`**, split on sentence-final punctuation `.!?` (runs count as one
   boundary), tokenize each sentence, and drop empty sentences. The boundary punctuation
   is **not** kept as a token.

### Task 2: The n-gram model (70 pts)
Complete the `NGramLM` class:
3. **`pad(tokens)`**, add `n−1` `<s>` markers in front and one `</s>` marker at the end.
4. **`ngrams(tokens)`**, slide a length-`n` window over a (padded) token list, returning
   tuples.
5. **`fit(corpus)`**, build the vocabulary (all observed tokens **plus** `</s>` and
   `<unk>`, but **not** `<s>`), then tally `ngram_counts` and `context_counts`.
6. **`prob(token, context)`**, add-k smoothed `P(token | context)`. Map any
   out-of-vocabulary token (in the target **or** the context) to `<unk>`.
7. **`sentence_logprob(tokens)`**, natural-log probability of one sentence (sum of
   per-token log-probabilities over the padded sequence).
8. **`perplexity(corpus)`**, `exp(− total_logprob / N)`, where `N` is the number of
   predicted tokens (n-grams scored, including the `</s>` predictions).
9. **`generate(max_len, seed)`**, sample a sentence from the model, starting at the
   `(<s>,…)` context and stopping at `</s>` or `max_len`. Must be **reproducible** given a
   `seed` (use a local `random.Random(seed)`). Do not return `<s>`/`</s>`.

### Task 3: Short written reflection (15 pts)
Add a top-of-file docstring or a `REFLECTION.md` (≤ 250 words) answering:
- **(a)** Train a bigram model on the provided corpus, then increase `k` from `0.01` to
  `1.0`. What happens to training perplexity, and **why**?
- **(b)** Why does a higher-order model (trigram vs. bigram) tend to have *lower* training
  perplexity but risk *higher* perplexity on unseen text?
- **(c)** Name one concrete failure mode of add-k smoothing and one alternative smoothing
  method that addresses it (cite J&M Ch. 3).

## Deliverables
- A completed `ngram_lm.py` that passes the public tests.
- Your written reflection (Task 3).
- The AI-use disclosure (see below).

## Grading rubric (100 pts)
| Component | Points |
|---|---:|
| Task 1: `tokenize`, `sentences` | 15 |
| Task 2: padding & n-grams (`pad`, `ngrams`) | 10 |
| Task 2: `fit` (vocab + counts correct) | 15 |
| Task 2: `prob` (add-k, sums to 1, UNK handling) | 20 |
| Task 2: `sentence_logprob` + `perplexity` | 15 |
| Task 2: `generate` (reproducible, well-formed) | 10 |
| Task 3: written reflection | 15 |
| **Total** | **100** |

Partial credit is awarded per passing test. Code that does not run earns at most the
written-reflection points.

## How to run & test
All code must run in the course Docker image (CPU-only, no network):

```bash
# From the repository root.
# Run the public tests against YOUR code:
docker compose -f docker/docker-compose.yml run --rm course \
    python -m pytest homeworks/hw1 -q

# (Instructor / self-check) Run the same tests against the reference solution:
docker compose -f docker/docker-compose.yml run --rm \
    -e NGRAM_FROM=solution course \
    python -m pytest homeworks/hw1 -q
```

Before you implement anything, the suite **skips** (this is expected). As you fill in the
TODOs, tests turn from skipped to passing. Target: **20/20 passing**.

## AI-use disclosure (required)
Per the syllabus AI-use policy, you may use LLM tools as coding assistants, but you must:
**(a)** disclose any AI assistance at the top of your submission (which tool, for what),
**(b)** be able to explain **every line** of code you submit, and **(c)** write the Task 3
reflection in your own words. Add a short `AI-USE:` note in your file header. Undisclosed
AI use is an academic-integrity violation.
