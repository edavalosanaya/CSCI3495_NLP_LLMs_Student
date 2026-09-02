#!/usr/bin/env python3
"""W3C1 reference solution, a tiny TF-IDF search engine."""
from __future__ import annotations
import math
import re

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
    tokenized = []
    for d in docs:
        tokenized.append(tokenize(d))

    n = len(docs)

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

    return {"docs": tokenized, "n": n, "df": df, "idf": idf}


def count_terms(tokens: list[str]) -> dict:
    """How many times each term appears in this one document. That count is tf."""
    counts = {}
    for term in tokens:
        if term in counts:
            counts[term] = counts[term] + 1
        else:
            counts[term] = 1
    return counts


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


def magnitude(vec: dict) -> float:
    """The length of a sparse vector: the square root of its squared weights."""
    total = 0.0
    for weight in vec.values():
        total = total + weight * weight
    return math.sqrt(total)


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
