# W3C1 Walkthrough: TF-IDF search, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `search.py` in this folder. Every code block below is taken
from it, and every printed value was produced by running it on the shipped
`DOCS` (8 documents).

---

## Step 1, Build the index

**The idea.** Two passes' worth of counting, though it fits in one loop: which
documents contain each term (df), and the resulting idf weight.

```python
    for toks in tokenized:
        for term in set(toks):        # set(): count each DOCUMENT once
            df[term] += 1
    idf = {term: math.log(n / df[term]) for term in df}
    return {"docs": tokenized, "n": n, "df": dict(df), "idf": idf}
```

**`set(toks)` is the load-bearing detail.** Document frequency counts documents,
not occurrences. Without the `set`, a document saying "the the the" would push
df("the") to 3 by itself, idf would go negative, and every weight downstream
would be wrong in a way that still runs.

**Why idf is a log at all.** Without it, a term in 1 of 1000 documents would be
1000 times more important than a term in every document. The log compresses that
into a range where terms can actually be compared: over our 8 documents, idf runs
from 0.0 ("the", in 5 documents, gives log(8/5) = 0.47) up to 2.08 ("pizza", in
1). Note that a term in *every* document gets exactly log(1) = 0.

**The index stores tokenized documents.** `search` re-derives each document's
vector on every query, which is fine at this scale and is the obvious thing to
cache (see the stretch goals).

**What you should see:**

```python
>>> idx = build_index(["the cat", "the dog", "the bird"])
>>> idx["idf"]["the"]                 # log(3/3)
0.0
>>> round(idx["idf"]["cat"], 4)       # log(3/1)
1.0986

>>> idx = build_index(DOCS)
>>> idx["n"], idx["df"]["dog"], round(idx["idf"]["dog"], 4)
(8, 2, 1.3863)
>>> idx["df"]["pizza"], round(idx["idf"]["pizza"], 4)
(1, 2.0794)
```

---

## Step 2, Weight the terms

```python
def tfidf_vector(index: dict, tokens: list[str]) -> dict:
    counts = Counter(tokens)
    return {
        term: tf * index["idf"].get(term, 0.0)
        for term, tf in counts.items()
        if index["idf"].get(term, 0.0) != 0.0
    }
```

**The `if` clause drops two different cases at once**, and it is worth naming
both:

1. Terms with **idf exactly 0** (present in every document). They carry no
   discriminating information, so a weight of 0 is not an approximation, it is
   the right answer.
2. Terms **not in the index at all** (a query word no document contains).
   `.get(term, 0.0)` gives them 0, and they are dropped for the same reason.

Keeping them as explicit `0.0` entries would give identical cosine scores. It
would just make the vectors dense, which defeats the point of a sparse
representation and slows the dot product down.

**What you should see:**

```python
>>> idx = build_index(["the cat", "the dog"])
>>> tfidf_vector(idx, ["the", "the", "cat"])
{'cat': 0.6931471805599453}
```

"the" appeared twice and is still absent. Students often expect `{'the': 0.0,
'cat': 0.69}` and are surprised; the test asserts `"the" not in vec` precisely to
force the conversation.

---

## Step 3, Compare two vectors

```python
def cosine(u: dict, v: dict) -> float:
    # Iterate over the smaller dict for the dot product.
    small, large = (u, v) if len(u) <= len(v) else (v, u)
    dot = sum(w * large.get(term, 0.0) for term, w in small.items())
    nu = math.sqrt(sum(w * w for w in u.values()))
    nv = math.sqrt(sum(w * w for w in v.values()))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)
```

**Only shared terms contribute to the dot product.** A term present in one vector
and missing from the other multiplies by 0, so iterating the smaller dict and
looking up in the larger gives the same answer as iterating the union, in less
time. This is the sparse-vector idiom worth internalizing; it is how real
inverted indexes work.

**The zero-norm guard is not paranoia.** An empty query vector happens constantly
in practice: search for a word no document contains, and every weight was
dropped in Step 2. Without the guard that is a `ZeroDivisionError` on a perfectly
ordinary query.

**Why divide by the norms at all.** The raw dot product rewards long documents,
which have more terms and therefore bigger numbers. Dividing by both norms
projects onto the unit sphere, so only the *direction* survives, which is why the
test asserts that `{a:1, b:2}` and `{a:3, b:6}` score 1.0.

**Floating point.** Identical vectors return `0.9999999999999998`, not `1.0`. The
tests use `pytest.approx` for this reason; a student comparing with `==` will see
a mystifying failure.

**What you should see:**

```python
>>> round(cosine({"a": 1.0}, {"a": 1.0}), 6)
1.0
>>> cosine({"a": 1.0}, {"b": 1.0})
0.0
>>> cosine({}, {"a": 1.0})
0.0
>>> round(cosine({"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 6.0}), 6)
1.0
```

---

## Step 4, Search

```python
def search(index: dict, query: str, k: int = 3) -> list[tuple[int, float]]:
    qvec = tfidf_vector(index, tokenize(query))
    scored = []
    for doc_id, toks in enumerate(index["docs"]):
        dvec = tfidf_vector(index, toks)
        scored.append((doc_id, cosine(qvec, dvec)))
    # Sort by score descending, then doc_id ascending.
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]
```

**The tie-break key.** `(-x[1], x[0])` sorts by score descending (negating flips
the direction) and by doc_id ascending within a tie. Doing it as two separate
sorts also works because Python's sort is stable, but the single key is clearer
about intent.

**Why the tie-break is tested at all.** Search for `"zzzznonexistent"` and every
document scores exactly 0.0. Without a defined tie-break the returned order
depends on dict iteration order, and the exercise would have a test that passes
or fails for reasons the student cannot see. Deterministic output is worth
insisting on; it is the same discipline as seeding an RNG.

**The obvious inefficiency**, worth naming so students see it: every query
recomputes every document vector. Real engines compute document vectors once at
index time and store them normalized, so a query is a sparse dot product against
a precomputed matrix. That is the second stretch goal.

**What you should see:**

```python
>>> idx = build_index(DOCS)
>>> search(idx, "hot dog mustard", k=1)
[(3, 0.5820...)]
>>> [d for d, _ in search(idx, "zzzznonexistent", k=3)]
[0, 1, 2]
```

---

## Step 5, Run the whole thing

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

**Three things to draw out in the debrief.**

1. **"cat and dog" vs "hot dog mustard" reverses documents 1 and 3.** Both
   queries contain "dog". The reversal comes entirely from "mustard" and
   "grilled" being rare, hence heavy. The engine has no concept of "hot dog" as a
   compound; it is doing weighted term overlap and nothing else.

2. **The third result is often 0.000.** It appears only because `k=3` was
   requested. A production engine cuts off at a relevance threshold. This is
   directly relevant to Week 11: a RAG system that always returns `k` chunks will
   hand the model irrelevant context, and the model will use it anyway.

3. **"team goal championship" ranks basketball (0.428) above soccer (0.390)**,
   even though the query contains "goal" and only the soccer document has it.
   Basketball wins on "team" plus "championship" against soccer's "team" plus
   "goal", and length normalization does the rest. Ask the class whether that is
   the right answer. There is no purely lexical way to decide, which is the
   opening for next class.

**The failure to name before students leave.** Synonyms. "couch" and "sofa" have
cosine 0 under this model, exactly as unrelated as "couch" and "championship",
because the representation has no notion that two different strings can mean the
same thing. Every term is its own orthogonal dimension. That is precisely the gap
dense embeddings close in W3C2, and it is worth writing on the board as the
question the next class answers.
