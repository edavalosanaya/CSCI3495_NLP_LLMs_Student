# HW2: Text Classification & Word Embeddings

**Out:** Week 4, Class 1 · **Due:** Week 5, Class 1 (start of class)
**100 points** · **Weight:** 2.5% of the course grade · **Individual assignment** · **Estimated time:** 5-7 hours

---

## Learning goals
By completing this homework you will be able to:
1. Implement a **Multinomial Naive Bayes** text classifier with add-1 smoothing **from
   scratch** in NumPy.
2. Evaluate a classifier with **precision, recall, and F1**.
3. Compute **cosine similarity** and find **nearest neighbors** in an embedding space.
4. Solve the classic **word analogy** task (`king − man + woman ≈ queen`) and run a simple
   **embedding-bias probe**.

## Background
This assignment spans Week 2 (text classification) and Week 3 (vector semantics & word
embeddings), corresponding to **Jurafsky & Martin Ch. 4 (Naive Bayes), Ch. 6 (vector
semantics & embeddings)**.

**Part A.** Multinomial Naive Bayes models a document as a bag of words and picks the class
that maximizes the posterior. Using Bayes' rule and the conditional-independence
assumption, with add-1 (Laplace) smoothing over vocabulary `V`:

> ĉ = argmax_c [ log P(c) + Σ_w count(w, doc) · log P(w | c) ]
> P(w | c) = (count(w, c) + 1) / (Σ_w' count(w', c) + |V|)

We work in **log space** to avoid floating-point underflow.

**Part B.** Word embeddings map words to dense vectors so that *distributional similarity*
becomes *geometric closeness*. This is the core idea behind **word2vec** (Mikolov et al.,
2013, "Efficient Estimation of Word Representations in Vector Space", arXiv:1301.3781) and
**GloVe** (Pennington, Socher & Manning, 2014, "GloVe: Global Vectors for Word
Representation", EMNLP). A striking property is **linear analogy structure**: vector
arithmetic like `king − man + woman` lands near `queen`. The same geometry that encodes
useful meaning also encodes **social bias** (e.g. Bolukbasi et al., 2016; Caliskan et al.,
2017 "WEAT"), which you will probe with a small association test.

You will work in **NumPy + the Python standard library**. For Part A's graded functions,
implement the math yourself, do **not** call scikit-learn's classifiers.

## Files

```
hw2/
  text_clf_embed.py        # <- YOU implement the TODOs here
  test_text_clf_embed.py   # the tests each step below refers to
  README.md                # this handout
```

## How this homework works

This handout is a sequence of steps. Each step is one function, and **each step
ends with a test you can run**, so you always know whether you are done before
you move on. Work them in order: later steps import earlier ones.

Open a shell inside the course image, already in this homework's folder.
One command, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/homeworks/hw2 course bash
```

Everything below runs in that shell:

```bash
pytest -k step3 -q      # check ONLY step 3
pytest -q               # run every step
```

**Before you write anything, every test skips.** That is expected: the suite
detects the unfinished starter and skips rather than drowning you in failures.
The moment step 1 is implemented the tests start running for real.

**Total when you are finished: `16 passed`.**

### Step 0, Orientation (0 pts)

Nothing to write yet.

Read `text_clf_embed.py` top to bottom. Note `_TOKEN_RE` at the top and the
attributes `NaiveBayesClassifier` promises in its docstring (`vocab_`, `classes_`,
`log_prior_`, `log_likelihood_`), because the tests check those names exactly. Then:

```bash
pytest -q
```

You should get `16 skipped`.

### Step 1, `tokenize` (5 pts)

**Write** `tokenize(text)`: lowercase, then apply `_TOKEN_RE`. Punctuation is dropped here, unlike HW1.

**Done when** `pytest -k step1 -q` prints `1 passed, 15 deselected`.

**Check it by hand**

```python
>>> from text_clf_embed import tokenize
>>> tokenize("Wow!! Great, great movie.")
['wow', 'great', 'great', 'movie']
```

**Why it matters.** Note that `great` appears twice. Naive Bayes is a *bag of words with counts*, not a set, and the repeat is real evidence. Deduplicating here quietly changes the model.

### Step 2, `NaiveBayesClassifier.fit` (20 pts)

**Write** `fit(docs, labels)`. Build the sorted vocabulary `vocab_`, the class list `classes_`, the log priors `log_prior_`, and `log_likelihood_[c]`, a NumPy array of `log P(w|c)` aligned with `vocab_`, using add-1 smoothing. For each class, `P(w|c)` over the vocabulary must sum to 1.

**Done when** `pytest -k step2 -q` prints `2 passed, 14 deselected`.

**Check it by hand**

```python
>>> docs = ["great movie", "great film", "awful movie", "awful boring film"]
>>> labels = ["pos", "pos", "neg", "neg"]
>>> clf = NaiveBayesClassifier().fit(docs, labels)
>>> sorted(clf.vocab_)
['awful', 'boring', 'film', 'great', 'movie']
>>> {k: round(v, 4) for k, v in clf.log_prior_.items()}
{'neg': -0.6931, 'pos': -0.6931}
>>> round(float(np.exp(clf.log_likelihood_['pos']).sum()), 6)
1.0
```

**Why it matters.** Two classes with two documents each gives `log(0.5) = -0.6931` for both priors. If your priors are not equal here, you are counting documents wrong before you even reach the likelihoods.

### Step 3, `_features`, `predict_log_scores`, `predict` (15 pts)

**Write** the three prediction methods. `_features` turns a document into a counts vector over `vocab_`; `predict_log_scores` returns `{class: log P(c) + counts . log P(w|c)}`; `predict` takes the argmax for each document. Words outside `vocab_` are simply ignored, not treated as errors.

**Done when** `pytest -k step3 -q` prints `3 passed, 13 deselected`.

**Check it by hand**

```python
>>> docs = ["great movie", "great film", "awful movie", "awful boring film"]
>>> labels = ["pos", "pos", "neg", "neg"]
>>> clf = NaiveBayesClassifier().fit(docs, labels)
>>> clf.predict(["great film", "awful movie"])
['pos', 'neg']
>>> clf.predict(["great film about aardvarks"])   # OOV word is ignored
['pos']
```

**Why it matters.** Working in log space turns the product over words into a sum, which is what keeps a 200-word document from underflowing to zero. It is the same reason HW1 summed logs.

### Step 4, `precision_recall_f1` (10 pts)

**Write** `precision_recall_f1(y_true, y_pred, positive)`. Return the three numbers for the given positive class. When the denominator is zero, return `0.0` rather than dividing.

**Done when** `pytest -k step4 -q` prints `2 passed, 14 deselected`.

**Check it by hand**

```python
>>> from text_clf_embed import precision_recall_f1
>>> y_true = ["pos", "pos", "neg", "neg"]
>>> y_pred = ["pos", "neg", "neg", "neg"]
>>> tuple(round(x, 4) for x in precision_recall_f1(y_true, y_pred, "pos"))
(1.0, 0.5, 0.6667)
```

**Why it matters.** Precision 1.0 with recall 0.5 is the classic shape of a cautious classifier: everything it flagged was right, and it missed half the real positives. Accuracy would have reported 0.75 and hidden that entirely.

### Step 5, `cosine_similarity` (8 pts)

**Write** `cosine_similarity(a, b)` as `dot(a, b) / (||a|| * ||b||)`, returning `0.0` if either vector is all zeros rather than dividing by zero.

**Done when** `pytest -k step5 -q` prints `2 passed, 14 deselected`.

**Check it by hand**

```python
>>> import numpy as np
>>> from text_clf_embed import cosine_similarity
>>> round(cosine_similarity(np.array([1., 0.]), np.array([1., 1.])), 4)
0.7071
>>> cosine_similarity(np.array([0., 0.]), np.array([1., 1.]))
0.0
```

**Why it matters.** 0.7071 is cos(45 degrees). Dividing by the norms is what makes this a measure of *direction*, so a long document and a short one about the same topic still score as similar.

### Step 6, `nearest_neighbors` (10 pts)

**Write** `nearest_neighbors(word, embeddings, k)`: the `k` most cosine-similar words, **excluding the query itself**, sorted by similarity descending with ties broken alphabetically.

**Done when** `pytest -k step6 -q` prints `2 passed, 14 deselected`.

**Check it by hand**

```python
>>> E = {"king": np.array([1., 1.]), "queen": np.array([1., 3.]),
...      "man": np.array([0., 0.]), "woman": np.array([0., 2.]),
...      "apple": np.array([-3., 0.5])}
>>> [(w, round(s, 4)) for w, s in nearest_neighbors("king", E, k=2)]
[('queen', 0.8944), ('woman', 0.7071)]
```

**Why it matters.** The tie-break is not pedantry: on real embeddings many pairs score identically to float precision, and without a deterministic rule your output changes between runs and the tests flicker.

### Step 7, `analogy` (12 pts)

**Write** `analogy(a, b, c, embeddings, k)`, solving *a is to b as c is to ?*. Build `v = emb[b] - emb[a] + emb[c]`, then return the `k` nearest words by cosine, **excluding a, b and c**.

**Done when** `pytest -k step7 -q` prints `2 passed, 14 deselected`.

**Check it by hand**

```python
>>> E = {"king": np.array([1., 1.]), "queen": np.array([1., 3.]),
...      "man": np.array([0., 0.]), "woman": np.array([0., 2.]),
...      "apple": np.array([-3., 0.5])}
>>> [(w, round(s, 4)) for w, s in analogy("man", "king", "woman", E, k=1)]
[('queen', 1.0)]
```

**Why it matters.** This is the Mikolov result you read about, reduced to arithmetic you can check by hand. The exclusion matters: without it the nearest vector to `king - man + woman` is usually `king` itself, and the demo collapses.

### Step 8, `bias_score` (5 pts)

**Write** `bias_score(word, group_a, group_b, embeddings)`: the mean cosine similarity of `word` to group A minus its mean similarity to group B. Positive means more associated with A.

**Done when** `pytest -k step8 -q` prints `2 passed, 14 deselected`.

**Check it by hand**

```python
>>> E = {"king": np.array([1., 1.]), "queen": np.array([1., 3.]),
...      "man": np.array([0., 0.]), "woman": np.array([0., 2.]),
...      "apple": np.array([-3., 0.5])}
>>> round(bias_score("king", ["man"], ["woman"], E), 4)
-0.7071
```

**Why it matters.** This is a miniature WEAT, the same shape as the published embedding-bias measures. The sign is the whole result, and it comes straight out of vectors nobody labelled for gender.

### Step 9, Run the whole thing (0 pts)

```bash
pytest -q
```

Every step green means `16 passed`. If a step you finished earlier has gone red,
you broke it with a later change; fix that before you submit.

## Written reflection (15 pts)

Answer in the module docstring or a short `REFLECTION.md`, a paragraph each:

1. Your Naive Bayes ignores word order entirely. Give a sentence pair it must classify
   identically but a human would not, and say what would be needed to fix that.
2. Step 8 gave you a number. What does it license you to claim, and what does it not?
   Be specific about what a single `bias_score` can and cannot show.
3. Naive Bayes assumes words are conditionally independent given the class, which is
   plainly false. Explain in your own words why the classifier still works.

## What to submit

- `text_clf_embed.py` with every TODO filled in and `pytest -q` fully green.
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
