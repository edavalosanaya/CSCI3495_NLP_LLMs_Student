# W3C1 Lab: Tiny Search Engine

## 1. Learning objective

Rank documents against a query using TF-IDF weights and cosine similarity, and
see what each of those two ideas contributes on its own.

You write two functions in `search.py`: the weighting and the similarity. The
index and the ranking loop are already written for you.

## 2. Understanding the math

![TF-IDF worked example: tf, df, idf, and the tf-idf product for four terms](../lecture/visuals/tfidf.png)

With $N$ documents and $\mathrm{df}_t$ the number of them containing term $t$,
a term is weighted by how often it occurs here against how rare it is overall:

$$\mathrm{idf}_t = \log\frac{N}{\mathrm{df}_t} \qquad\qquad w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t$$

A term in every document has $\mathrm{df}_t = N$, so its idf is $\log 1 = 0$ and
its weight vanishes no matter how often it appears.

![Cosine similarity: documents as vectors, ranked by angle, not length](../lecture/visuals/cosine-similarity.png)

Documents are compared by the angle between their weight vectors, not their
length, so a long document does not beat a short one just for being long:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-03/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `tfidf_vector`

Turn a token list into a sparse `{term: weight}` dict. Terms whose weight comes
out as zero are left out of the dict entirely.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 6 deselected
```

## 5. Implement `cosine`

Divide the dot product by the two vector lengths, and return `0.0` rather than
dividing by zero when a vector has no length.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 5 deselected
```

## 6. Run it, then break it

```bash
python search.py
```

```
============================================================
Tiny Search Engine, TF-IDF + cosine similarity
============================================================

query: 'cat and dog'
   0.447  [1] a dog and a cat can be good friends
   0.235  [3] i grilled a hot dog and ate it with mustard
   0.202  [0] the cat chased the mouse around the house

query: 'hot dog mustard'
   0.582  [3] i grilled a hot dog and ate it with mustard
   0.121  [1] a dog and a cat can be good friends
   0.000  [0] the cat chased the mouse around the house

query: 'team goal championship'
   0.428  [5] the basketball team won the championship game
   0.390  [4] the soccer team scored a last minute goal
   0.000  [0] the cat chased the mouse around the house
```

Each experiment below is a one-line edit; undo it before the next.

1. Turn idf off. In `build_index`, replace the `idf` dict with
   `{term: 1.0 for term in df}`, making the weight pure term frequency. The
   pizza document now appears third for `'cat and dog'` at 0.218. Which word
   put it there, and what was idf doing about that word before?
2. Make a term worthless. Prefix every document in `DOCS` with `the` so all 8
   contain it. `idf["the"]` becomes exactly 0.0 and `the` disappears from every
   vector. Is that the same thing as a stopword list, or different?
3. Try to spam the index. Add `"dog dog dog dog dog dog dog dog dog dog"` to
   `DOCS`. It reaches second for `'hot dog mustard'` at 0.333, not first. Which
   half of the formula stopped it, tf-idf or cosine?
4. Compare `cosine({"a": 1.0}, {"a": 100.0})` with
   `cosine({"a": 1.0}, {"b": 1.0})`. The first is 1.0 and the second 0.0. What
   does that say about what cosine actually measures?
