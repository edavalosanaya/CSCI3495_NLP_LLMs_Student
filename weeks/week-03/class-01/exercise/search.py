#!/usr/bin/env python3
"""W3C1 starter, a tiny TF-IDF search engine. See README.md."""
from __future__ import annotations
import math
import re
from collections import Counter

# Tiny one-line "documents". Topics overlap on purpose.
DOCS = [
    "the cat chased the mouse around the house",
    "a dog and a cat can be good friends",
    "fresh pizza with melted cheese and tomato",
    "i grilled a hot dog and ate it with mustard",
    "the soccer team scored a last minute goal",
    "the basketball team won the championship game",
    "she baked a chocolate cake for the party",
    "the puppy chased its tail in the yard",
]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer."""
    return re.findall(r"[a-z']+", text.lower())


def build_index(docs: list[str]) -> dict:
    """GIVEN. Tokenizes the corpus and computes df and idf for every term.

    Returns {"docs": tokens per doc, "n": doc count, "df": term -> doc
    frequency, "idf": term -> log(n / df)}.
    """
    tokenized = [tokenize(d) for d in docs]
    n = len(docs)
    df: Counter = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1
    idf = {term: math.log(n / df[term]) for term in df}
    return {"docs": tokenized, "n": n, "df": dict(df), "idf": idf}


def tfidf_vector(index: dict, tokens: list[str]) -> dict:
    """Weight one document (or query) as a sparse term -> tf-idf vector.

    Args:
        index: the dict from build_index. Only its "idf" mapping is needed
            here, and a term the corpus never contained is absent from it.
        tokens: the tokenized document or query being weighted. The same term
            may appear many times; that repetition is the tf.

    Returns:
        {term: weight} holding only terms that carry signal. A term whose idf
        is 0.0 is left out entirely rather than stored as zero.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The weight formula is in README section 2.
    #
    #   count how many times each token occurs -- that count is the tf
    #   look each term's idf up in the index, treating a missing term as 0.0
    #   keep only the terms whose weight is not zero
    #
    #   A term with idf 0.0 sits in every document, so it separates nothing.
    #
    raise NotImplementedError


def cosine(u: dict, v: dict) -> float:
    """Cosine of the angle between two sparse term -> weight vectors.

    Args:
        u: a sparse vector. Terms absent from it are zero, not missing data.
        v: the other sparse vector. It need not share any term with u.

    Returns:
        A float in [0, 1] for non-negative weights. 0.0 when either vector has
        zero length, since a vector with no length points nowhere and asking
        for its angle is meaningless.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The formula is in README section 2.
    #
    #   the dot product only needs the terms u actually has: a term missing
    #       from v contributes nothing
    #   each vector's length is the square root of the sum of its squared weights
    #   a zero length means there is no angle to measure, so give back 0.0
    #   otherwise divide the dot product by the two lengths
    #
    raise NotImplementedError


def search(index: dict, query: str, k: int = 3) -> list[tuple[int, float]]:
    """GIVEN. Ranks every document against the query, ties broken by doc_id."""
    qvec = tfidf_vector(index, tokenize(query))
    scored = []
    for doc_id, toks in enumerate(index["docs"]):
        dvec = tfidf_vector(index, toks)
        scored.append((doc_id, cosine(qvec, dvec)))
    # Sort by score descending, then doc_id ascending.
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]


def _demo() -> None:
    index = build_index(DOCS)
    queries = ["cat and dog", "hot dog mustard", "team goal championship"]
    print("=" * 60)
    print("Tiny Search Engine, TF-IDF + cosine similarity")
    print("=" * 60)
    for q in queries:
        print(f"\nquery: {q!r}")
        for doc_id, score in search(index, q, k=3):
            print(f"   {score:5.3f}  [{doc_id}] {DOCS[doc_id]}")


if __name__ == "__main__":
    try:
        _demo()
    except NotImplementedError:
        print("search.py is not finished yet: fill in the next TODO in this file, then re-run.")
