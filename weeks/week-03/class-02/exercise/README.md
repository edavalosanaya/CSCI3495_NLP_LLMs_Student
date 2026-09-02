# W3C2 Lab: Word Embeddings, Analogies & Bias

## 1. Learning objective

Do arithmetic on word vectors: solve analogies with a vector offset, then use
the same offset trick to measure how a word leans along a social axis.

You write two functions in `embeddings.py`. Cosine similarity and the
nearest-neighbour search are already written for you.

## 2. Understanding the math

![Analogy as vector arithmetic: the man to woman offset is parallel to the king to queen offset](../lecture/visuals/analogy-vectors.png)

Similarity is the angle between two vectors:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$$

"$a$ is to $b$ as $c$ is to ?" becomes a step in vector space: take the offset
from $a$ to $b$, apply it at $c$, and look for the nearest word to where you land:

$$\text{analogy}(a, b, c) = \arg\max_{w \notin \{a, b, c\}} \; \cos\big(\mathrm{vec}(w), \; \mathrm{vec}(b) - \mathrm{vec}(a) + \mathrm{vec}(c)\big)$$

![Occupations projected onto the man to woman direction, after Bolukbasi et al. 2016](../lecture/visuals/embedding-bias.png)

The same offset defines an axis. Projecting a word onto it says which end that
word leans toward, which is how Bolukbasi et al. (2016) measured occupational
gender bias in real embeddings:

$$\mathrm{bias}(w) = \cos\big(\mathrm{vec}(w), \; \mathrm{vec}(\mathit{pos}) - \mathrm{vec}(\mathit{neg})\big)$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-03/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `analogy`

Build the target vector, score every word against it, and return the best `k`
excluding `a`, `b` and `c`. Break ties by the word itself so the output is
deterministic.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 5. Implement `bias_score`

Project one word onto the axis running from `neg` to `pos`. Two lines.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 6. Run it, then question it

```bash
python embeddings.py
```

```
============================================================
Word Embeddings, neighbors, analogies & bias probing
============================================================

Nearest neighbors of 'king':
   0.927  prince
   0.747  queen
   0.700  uncle

Analogy  man : king :: woman : ?
   0.996  queen

Bias probe along the (woman - man) direction:
   nurse     +0.343  -> woman
   doctor    -0.036  -> man
   engineer  -0.340  -> man
   teacher   +0.226  -> woman

(Illustrative toy vectors; real embeddings show the same patterns.)
```

These are toy vectors, built to show the pattern. The questions are the point.

1. Run the analogy backwards: `analogy("woman", "queen", "man", EMB, k=1)`. It
   returns `king` with similarity 0.9958, exactly the score the forward
   direction gave. Look at the target-vector formula and explain why the two
   must match.
2. Flip the axis. Compare `bias_score("nurse", "she", "he", ...)` against
   `bias_score("nurse", "he", "she", ...)` using `woman`/`man`: +0.343 becomes
   -0.343. Which part of the formula forces an exact sign flip?
3. Probe a word that is gendered by definition: `bias_score("queen", "woman",
   "man", EMB)` is +0.355, about the same as `nurse` at +0.343. Both score
   alike, but only one of them is evidence of a problem. What distinguishes
   them, and what does that mean for using this number as a bias metric?
4. Delete the "exclude a, b, c" guard from `analogy`. The top answer is still
   `queen`, unchanged. Construct a case where dropping the guard would change
   the answer, and say what that implies about how far apart these toy vectors
   are.
