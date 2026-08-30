#!/usr/bin/env python3
"""W3C1 reference solution, a tiny TF-IDF search engine."""
from __future__ import annotations
import math
import re
from collections import Counter

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
    return re.findall(r"[a-z']+", text.lower())


def build_index(docs: list[str]) -> dict:
    tokenized = [tokenize(d) for d in docs]
    n = len(docs)
    df: Counter = Counter()
    for toks in tokenized:
        for term in set(toks):
            df[term] += 1
    idf = {term: math.log(n / df[term]) for term in df}
    return {"docs": tokenized, "n": n, "df": dict(df), "idf": idf}


def tfidf_vector(index: dict, tokens: list[str]) -> dict:
    counts = Counter(tokens)
    return {
        term: tf * index["idf"].get(term, 0.0)
        for term, tf in counts.items()
        if index["idf"].get(term, 0.0) != 0.0
    }


def cosine(u: dict, v: dict) -> float:
    # Iterate over the smaller dict for the dot product.
    small, large = (u, v) if len(u) <= len(v) else (v, u)
    dot = sum(w * large.get(term, 0.0) for term, w in small.items())
    nu = math.sqrt(sum(w * w for w in u.values()))
    nv = math.sqrt(sum(w * w for w in v.values()))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def search(index: dict, query: str, k: int = 3) -> list[tuple[int, float]]:
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
    _demo()
