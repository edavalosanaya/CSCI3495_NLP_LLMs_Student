# W3C2 Walkthrough: Word embeddings, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `embeddings.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it on the shipped
toy `EMB` table (15 words, 8 dimensions).

**A standing caveat to repeat in class.** `EMB` is hand-built, not trained. Its
dimensions are labelled (`0=royal, 1=masculine, 2=feminine, ...`) so that the
vector arithmetic has something clean to find. Real word2vec and GloVe
dimensions mean nothing individually. What transfers is the *math* you write
here, and the fact that the same probes find the same patterns in real vectors.

---

## Step 1, Cosine similarity

**The idea.** Same formula as last class, now on dense numpy arrays instead of
sparse dicts.

```python
def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))
```

**Why this is a one-liner now.** In W3C1 the vectors were sparse dicts, so the
dot product needed a loop over shared keys. Here every word has a value in every
dimension, so `np.dot` does the whole thing. Worth pausing on: this is the
sparse-to-dense transition the entire class is about, and it shows up first in
how much simpler the code gets.

**The zero-norm guard survives** for the same reason as before: `np.dot` would
happily return 0 and then divide by 0, giving `nan`, which silently poisons every
downstream comparison rather than raising.

**The `float(...)` casts** keep the return type a Python float rather than a
`np.float64`. Not required, but it stops surprising repr differences in test
output.

**What you should see:**

```python
>>> round(cosine(vec("king"), vec("king")), 6)
1.0
>>> round(cosine(vec("king"), vec("queen")), 4)
0.7474
>>> round(cosine(vec("king"), vec("cat")), 4)
0.0
```

`king` and `cat` are exactly orthogonal here because the toy table puts royalty
and animals on disjoint dimensions. Real embeddings never give a clean 0; they
give small positive noise, because every word co-occurs with every other word
occasionally.

---

## Step 2, Nearest neighbors

```python
def nearest(word: str, table: dict, k: int = 3) -> list[tuple[str, float]]:
    target = np.asarray(table[word], dtype=float)
    scored = [
        (w, cosine(target, np.asarray(v, dtype=float)))
        for w, v in table.items()
        if w != word
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]
```

**`if w != word` is the whole exercise.** Every word is its own nearest neighbor
at cosine 1.0, so without the exclusion the top result is always the query and
the function is useless. The test checks this explicitly.

**The sort key `(-x[1], x[0])`** is descending by similarity, then alphabetical
by word. The tie-break is not decorative here: several words in the toy table
score exactly 0.0 against a given query, and without it the returned order would
depend on dict iteration order. Same determinism discipline as W3C1's doc-id
tie-break.

**What you should see:**

```python
>>> [(w, round(s, 4)) for w, s in nearest("cat", EMB, k=3)]
[('kitten', 1.0), ('dog', 0.994), ('aunt', 0.0)]
>>> [(w, round(s, 4)) for w, s in nearest("king", EMB, k=3)]
[('prince', 0.927), ('queen', 0.7474), ('uncle', 0.7001)]
```

**Two things worth pointing at.**

- `kitten` scores exactly **1.0** with `cat`. In the toy table their vectors are
  parallel (`[0,0,0,0.9,0.8,...]` and `[0,0,0,0.95,0.85,...]` differ only in
  length), and cosine ignores length. This is a good, concrete reminder that
  cosine measures direction only.
- The third neighbor of `cat` is `aunt` at **0.0**. There is no third animal in
  the table, so `nearest` is scraping the bottom and returning something
  unrelated. Same lesson as the 0.000 result in W3C1's search: a fixed `k`
  always returns `k` things, whether or not `k` things are relevant.

---

## Step 3, Analogies

**The idea.** `man : king :: woman : ?` becomes vector arithmetic. Take the
king-minus-man offset (roughly "royalty"), add it to woman, and look for the
nearest word to the result.

```python
def analogy(a: str, b: str, c: str, table: dict, k: int = 1) -> list[tuple[str, float]]:
    target = (
        np.asarray(table[b], dtype=float)
        - np.asarray(table[a], dtype=float)
        + np.asarray(table[c], dtype=float)
    )
    exclude = {a, b, c}
    scored = [
        (w, cosine(target, np.asarray(v, dtype=float)))
        for w, v in table.items()
        if w not in exclude
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]
```

**The exclusion set is not a technicality.** Without it, `analogy("man", "king",
"woman")` returns **`king`**, not `queen`. The target vector is
`king - man + woman`, and since `man` and `woman` are nearly orthogonal in this
space, the result stays closest to `king` itself. Every published word2vec
analogy result excludes the three input words for exactly this reason, and it is
a fair criticism of how impressive those results were made to look. Say that out
loud; students find it clarifying rather than deflating.

**What you should see:**

```python
>>> [(w, round(s, 4)) for w, s in analogy("man", "king", "woman", EMB, k=1)]
[('queen', 0.9958)]
```

0.996 is suspiciously good because the toy table was built with a consistent
gender offset. On real GloVe vectors the same query lands on `queen` at roughly
0.7, and many analogies in the standard test sets simply fail.

---

## Step 4, Bias probing

```python
def bias_score(word: str, pos: str, neg: str, table: dict) -> float:
    direction = np.asarray(table[pos], dtype=float) - np.asarray(table[neg], dtype=float)
    return cosine(np.asarray(table[word], dtype=float), direction)
```

**Reading the formula.** `vec(woman) - vec(man)` is a direction in the space, the
axis along which those two words differ. Projecting a third word onto it asks
"which end of this axis does this word sit on?" Positive means it leans toward
`pos`, negative toward `neg`, and the magnitude says how strongly.

This is a simplified version of the WEAT / Bolukbasi methodology. Real work
averages over several word pairs to define the direction, because a single pair
is noisy and carries its own idiosyncrasies.

**What you should see:**

```python
>>> for occ in ["nurse", "doctor", "engineer", "teacher"]:
...     print(occ, round(bias_score(occ, "woman", "man", EMB), 4))
nurse 0.3434
doctor -0.0357
engineer -0.3402
teacher 0.2265
```

**How to teach this honestly.** These numbers come from a table a human wrote, so
they prove nothing on their own; the association was put there deliberately, in
the `5=care-work` and `6=technical` dimensions. What matters is that the *probe*
is real, and that running the identical probe on genuine word2vec vectors trained
on Google News produces the same shape of result, which is the finding in
Bolukbasi et al. 2016 and Caliskan et al. 2017.

The point for students is not "embeddings are biased" as a slogan. It is:

1. A representation learned from text absorbs the statistical regularities of
   that text, including social ones. Nobody put them there on purpose.
2. Those regularities are *measurable*, with four lines of arithmetic they just
   wrote.
3. A downstream system (resume screening, search ranking) inherits them silently,
   because the embedding is a component nobody inspects.

Note `doctor` sits at roughly **-0.04**, close to neutral, while `nurse` and
`engineer` sit near the extremes. That asymmetry is worth asking about: what
would it mean for a real corpus to place one occupation near zero and another far
out?

---

## Step 5, Run the whole thing

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

**Closing the two-class arc.** W3C1 ended on a specific failure: "couch" and
"sofa" have cosine 0 under TF-IDF, exactly as unrelated as "couch" and
"championship", because every term is its own orthogonal dimension. Dense
embeddings fix precisely that, and `nearest` is the demonstration: `kitten` and
`cat` are close without sharing a single character.

What the students should also leave with is what embeddings did **not** fix.
Every vector here is static: `bank` gets one vector whether the sentence is about
rivers or money. That is the gap Week 6 opens with ELMo and contextual
representations, and it is worth writing on the board next to the synonym
question from last class.

## On the stretch goal

`load_pretrained()` tries `sentence-transformers` and returns `None` with a clear
message if the model is not cached or the machine is offline. If a student gets
real vectors loaded, the interesting comparison is not that the analogies work
better; it is that they work *less* cleanly than the toy table, with analogy
scores around 0.7 and a meaningful failure rate. That gap between the tidy
classroom example and the real artifact is worth more than another passing test.
