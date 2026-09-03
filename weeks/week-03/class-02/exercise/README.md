# W3C2 Lab: Word Embeddings, Analogies & Bias

## 1. Learning objective

Do arithmetic on word vectors: solve analogies with a vector offset, then use
the same offset trick to measure how a word leans along a social axis.

You write two functions in `embeddings.py`. Cosine similarity and the
nearest-neighbour search are already written for you.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-03/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `analogy`

![Analogy as vector arithmetic: the man to woman offset is parallel to the king to queen offset](../lecture/visuals/analogy-vectors.png)

Similarity is the angle between two vectors, which the given `cosine` measures:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$$

"$a$ is to $b$ as $c$ is to ?" becomes a step in vector space: take the offset
from $a$ to $b$, apply it at $c$, and look for the nearest word to where you land:

$$\text{analogy}(a, b, c) = \arg\max_{w \notin \{a, b, c\}} \; \cos\big(\mathrm{vec}(w), \; \mathrm{vec}(b) - \mathrm{vec}(a) + \mathrm{vec}(c)\big)$$

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

## 4. Implement `bias_score`

![Occupations projected onto the man to woman direction, after Bolukbasi et al. 2016](../lecture/visuals/embedding-bias.png)

The same offset defines an axis. Projecting a word onto it says which end that
word leans toward, which is how Bolukbasi et al. (2016) measured occupational
gender bias in real embeddings:

$$\mathrm{bias}(w) = \cos\big(\mathrm{vec}(w), \; \mathrm{vec}(\mathit{pos}) - \mathrm{vec}(\mathit{neg})\big)$$

Project one word onto the axis running from `neg` to `pos`. Two lines.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 6 deselected
```

## 5. Run it, then discuss it

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

These are toy vectors, built to show the pattern. Everything below is a
discussion, run in pairs or small groups: make the edit, look at the number,
then argue about what it means. Undo each edit before starting the next.

1. Nobody trained these vectors, someone typed them. Find the line in `EMB`
   that makes `nurse` lean toward `woman`, and the line that makes `engineer`
   lean toward `man`. In a real embedding, no one types those numbers. Where
   would the same lean come from instead, and would anyone have noticed it?

2. Try the obvious fix. In `EMB`, set the third number of `nurse` (its feminine
   coordinate) to `0.0` and re-run. `bias_score` now reports `+0.000`. Then look
   at where `nurse` actually sits:

   ```
   nurse: doctor 0.908, teacher 0.801, engineer 0.096
   ```

   The metric says the bias is gone. Would a resume ranker built on these
   vectors behave any differently? Decide as a group whether you fixed the
   problem or just blinded the measurement.

3. Now patch all four: zero the feminine coordinate of `nurse` and `teacher`,
   and the masculine coordinate of `doctor` and `engineer`. `nurse` and
   `engineer` go to `+0.000`, but `doctor` climbs to `+0.187` and `teacher`
   moves to `-0.129`. You edited every one of the four and two of them still
   lean, because one coordinate is not the whole axis. Propose a rule that
   would not need patching word by word.

4. Count the cost. Restore `EMB`, then zero the feminine coordinate of `queen`
   and re-run the analogy: the top answer becomes `princess` at 0.935 and
   `queen` falls to second at 0.831. The gender information you removed is the
   same information Step 1 depends on. Where is the line between a stereotype
   and a fact about a word, and who on your team gets to draw it?

5. Pick your point of intervention: change the training data, edit the vectors
   afterward, or leave the vectors alone and restrict what they may be used
   for. Choose one as a group, defend it, and name what it costs. Be ready to
   say which of the four experiments above changed your mind.
