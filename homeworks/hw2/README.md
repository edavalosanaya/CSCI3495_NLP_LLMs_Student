# HW2: Text Classification & Word Embeddings

**Out:** Week 4, Class 1 · **Due:** Week 5, Class 1 (start of class)
**Weight:** 5% of course grade · **Individual assignment** · **Estimated time:** 5-7 hours

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
  test_text_clf_embed.py   # public tests
  README.md                # this handout
```

## Tasks

### Part A: Naive Bayes & evaluation (50 pts)
1. **`tokenize(text)`**, lowercase, split into `[a-z0-9']+` tokens.
2. **`NaiveBayesClassifier.fit(docs, labels)`**, learn sorted `classes_`, sorted `vocab_`,
   `log_prior_` per class, and `log_likelihood_[c]` (a NumPy array of `log P(w|c)` aligned
   with `vocab_`) using add-1 smoothing. P(w|c) over the vocabulary must sum to 1 per class.
3. **`predict_log_scores` / `predict`**, score each class in log space and return the
   argmax. Out-of-vocabulary words at test time are ignored.
4. **`precision_recall_f1(y_true, y_pred, positive)`**, binary P/R/F1 (return `0.0` for any
   zero-denominator case).

### Part B: Word embeddings (35 pts)
5. **`cosine_similarity(a, b)`**, return `0.0` if either vector is the zero vector.
6. **`nearest_neighbors(word, embeddings, k)`**, top-`k` words by cosine similarity,
   excluding the query word; sort by similarity descending, ties broken alphabetically.
7. **`analogy(a, b, c, embeddings, k)`**, compute `emb[b] − emb[a] + emb[c]` and return the
   nearest word(s), **excluding** `a`, `b`, `c`.
8. **`bias_score(word, group_a, group_b, embeddings)`**, mean cosine similarity to group A
   minus mean to group B (a mini-WEAT association score).

### Part C: Short written reflection (15 pts)
In a `REFLECTION.md` (≤ 250 words) answer:
- **(a)** Why does Naive Bayes work *well* on this small, lexically-separable dataset, and
  what is one realistic text domain where its independence assumption hurts it?
- **(b)** Explain *why* analogies can be solved by vector arithmetic, what does the
  difference `king − man` approximately represent?
- **(c)** Your `bias_score` for a real embedding set might show, say, "nurse" closer to
  female words. Name one concrete downstream harm if such embeddings feed a hiring model,
  and one mitigation.

## Deliverables
- Completed `text_clf_embed.py` passing the public tests.
- `REFLECTION.md` (Part C).
- AI-use disclosure (see below).

## Grading rubric (100 pts)
| Component | Points |
|---|---:|
| `tokenize` | 5 |
| `NaiveBayesClassifier.fit` (vocab, priors, smoothed likelihoods) | 20 |
| `predict_log_scores` / `predict` (correct argmax, OOV handling) | 15 |
| `precision_recall_f1` | 10 |
| `cosine_similarity` (incl. zero-vector guard) | 8 |
| `nearest_neighbors` (ordering + tie-breaks) | 10 |
| `analogy` (correct vector math + exclusions) | 12 |
| `bias_score` | 5 |
| Reflection (Part C) | 15 |
| **Total** | **100** |

## How to run & test
All code runs in the course Docker image (CPU-only, no network):

```bash
# Run the public tests against YOUR code:
docker compose -f docker/docker-compose.yml run --rm course \
    python -m pytest homeworks/hw2 -q

# (Instructor / self-check) against the reference solution:
docker compose -f docker/docker-compose.yml run --rm \
    -e HW2_FROM=solution course \
    python -m pytest homeworks/hw2 -q
```

Before you implement anything the suite **skips** (expected). Target: **14/14 passing**.

## AI-use disclosure (required)
Per the syllabus AI-use policy: **(a)** disclose any AI assistance in your file header
(which tool, for what), **(b)** be able to explain every line you submit, and **(c)** write
the reflection in your own words. Add an `AI-USE:` note in your header. Undisclosed AI use
is an academic-integrity violation.
