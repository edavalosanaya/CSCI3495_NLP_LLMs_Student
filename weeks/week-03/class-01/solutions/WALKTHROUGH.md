# W3C1 Walkthrough: TF-IDF search, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `search.py` in this folder. Every code block below is
copied from it. The code is written the long way on purpose: plain loops, one
idea per line, and a name for every intermediate value. Shorter versions exist,
but they hide the thing you are trying to learn.

---

## Given, `build_index`

You do not write this, but read it once: everything downstream depends on the
shape it returns.

```python
    # df[term] = how many documents contain the term at least once.
    df = {}
    for toks in tokenized:
        seen_in_this_doc = set(toks)
        for term in seen_in_this_doc:
            if term in df:
                df[term] = df[term] + 1
            else:
                df[term] = 1

    idf = {}
    for term in df:
        idf[term] = math.log(n / df[term])
```

**The `set(toks)` is the whole trick.** df counts DOCUMENTS, not occurrences.
A document that says "cat" nine times still adds exactly 1 to `df["cat"]`,
and wrapping the token list in a set is what enforces that.

If a term is in every document, `df[term] == n`, so `math.log(n / n)` is
`math.log(1)`, which is `0.0`. Remember that number; it comes back in step 1.

---

## Given, `count_terms`

```python
def count_terms(tokens: list[str]) -> dict:
    """How many times each term appears in this one document. That count is tf."""
    counts = {}
    for term in tokens:
        if term in counts:
            counts[term] = counts[term] + 1
        else:
            counts[term] = 1
    return counts
```

This is the standard "count things into a dictionary" loop. `collections.Counter`
does the same job in one line, and you should use it in real code, but written
out like this you can see there is no magic: look, add one or start at one.

---

## Step 1, `tfidf_vector`

**The idea.** Each term gets a weight: how often it appears HERE, times how
rare it is EVERYWHERE. Common words score low no matter how often they appear.

```python
def tfidf_vector(index: dict, tokens: list[str]) -> dict:
    counts = count_terms(tokens)

    weights = {}
    for term in counts:
        tf = counts[term]
        idf = index["idf"].get(term, 0.0)
        weight = tf * idf
        if weight == 0.0:
            # Either the corpus never had this term, or it is in every
            # document. Both mean it separates nothing, so leave it out.
            continue
        weights[term] = weight

    return weights
```

Read the loop body as four separate questions, which is why each gets its own
line and its own name:

1. `tf` — how many times is this term in this document?
2. `idf` — how rare is it across the corpus? `.get(term, 0.0)` because a query
   word that appears in no document at all is simply missing from the index.
3. `weight` — the two multiplied.
4. Is the weight zero? Then skip it.

**That `continue` drops two different cases at once**, and it is worth naming
both:

- Terms with **idf exactly 0**, meaning they are in every document. They
  separate nothing, so 0 is not an approximation, it is the right answer.
- Terms **not in the index at all**. `.get` gives them 0.0 for a different
  reason, and they are dropped the same way.

Storing them as explicit `0.0` entries would give identical cosine scores. It
would just make the vectors dense, which defeats the point of a sparse
representation.

**What you should see:**

```python
>>> idx = build_index(["the cat", "the dog"])
>>> tfidf_vector(idx, ["the", "the", "cat"])
{'cat': 0.6931471805599453}
```

"the" appeared twice and is still absent. Most people expect
`{'the': 0.0, 'cat': 0.69}`; the test asserts `"the" not in vec` precisely to
force that conversation.

**Common mistakes.** Using `index["df"]` instead of `index["idf"]`: both are
dicts keyed by term, so nothing crashes, and the ranking quietly goes backwards
because common words now score HIGHEST. Indexing with `index["idf"][term]`
instead of `.get`: correct for document terms, `KeyError` the first time a
query contains a word the corpus never had.

---

## Given, `magnitude`

```python
def magnitude(vec: dict) -> float:
    """The length of a sparse vector: the square root of its squared weights."""
    total = 0.0
    for weight in vec.values():
        total = total + weight * weight
    return math.sqrt(total)
```

Pythagoras, with as many dimensions as the vector has terms. Pulling it out as
its own function keeps `cosine` down to the three ideas that matter.

---

## Step 2, `cosine`

**The idea.** Two documents are similar if their weight vectors point the same
direction. Dividing by both lengths is what makes it about DIRECTION and not
about how long the documents are.

```python
def cosine(u: dict, v: dict) -> float:
    # A term missing from v contributes nothing, so only u's terms matter.
    dot = 0.0
    for term in u:
        if term in v:
            dot = dot + u[term] * v[term]

    u_length = magnitude(u)
    v_length = magnitude(v)

    if u_length == 0.0 or v_length == 0.0:
        return 0.0

    return dot / (u_length * v_length)
```

Three steps, in order:

1. **The dot product.** Walk `u`'s terms. A term `v` does not have contributes
   `weight * 0`, which is 0, so skipping it entirely gives the same answer with
   less work. That is why the loop is over `u` and not over every term in the
   vocabulary.
2. **The two lengths**, each from the helper above.
3. **The guard, then the division.** An empty vector has length 0, and dividing
   by it raises `ZeroDivisionError`. This is a real case, not a theoretical
   one: it happens whenever a query contains nothing the corpus has ever seen.

**What you should see:**

```python
>>> cosine({"a": 1.0}, {"a": 100.0})
1.0
>>> cosine({"a": 1.0}, {"b": 1.0})
0.0
```

The first is 1.0 because both vectors point along the same axis, however far
along it they go. The second is 0.0 because they share no terms, so the dot
product never gets a single non-zero contribution.

**Common mistakes.** Returning the dot product without dividing, which makes
long documents win everything. Checking `if u_length == 0` but not `v_length`,
which crashes on exactly half the bad inputs and therefore looks fine in
testing.

---

## Given, `search`

```python
def search(index: dict, query: str, k: int = 3) -> list[tuple[int, float]]:
    query_vec = tfidf_vector(index, tokenize(query))

    scored = []
    for doc_id, toks in enumerate(index["docs"]):
        doc_vec = tfidf_vector(index, toks)
        score = cosine(query_vec, doc_vec)
        scored.append((doc_id, score))

    # Highest score first; when two tie, the lower doc_id comes first.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]
```

The query is weighted with exactly the same function as the documents, using
the same index. That is what puts them in the same space, and it is why a query
term the corpus has never seen simply drops out instead of causing an error.

The sort key is a pair. Python compares pairs left to right, so `-score` orders
by score descending, and `doc_id` breaks ties ascending. Negating the score is
the usual way to sort one field down and another up in a single pass.

---

## Running it

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

The `0.000` rows are the interesting ones. `search` always returns `k` results,
even when nothing matches, because cosine always ranks something first. A real
search engine adds a score threshold below which it reports "no results" rather
than showing its three least-bad guesses.
