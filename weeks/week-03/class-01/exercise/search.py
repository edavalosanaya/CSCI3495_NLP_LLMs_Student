#!/usr/bin/env python3
"""W3C1 starter, a tiny TF-IDF search engine.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-03/class-01/exercise/test_search.py -k step1 -q

When all four steps are done, the demo runs:
    python weeks/week-03/class-01/exercise/search.py

CPU-only, deterministic, no network. Build TF-IDF and cosine yourself.
"""
from __future__ import annotations
import math
import re

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
    """Build a search index over the documents.

    Return a dict like:
        {
          "docs": [tokens_per_doc, ...],  # tokenized documents
          "n": number_of_docs,
          "df": {term: doc_frequency},    # # docs containing the term
          "idf": {term: log(n / df)},     # inverse document frequency
        }
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    # df counts DOCUMENTS containing the term, not total occurrences.
    raise NotImplementedError


def tfidf_vector(index: dict, tokens: list[str]) -> dict:
    """Return {term: tf-idf weight} for a list of tokens.

    tf = raw count of the term in `tokens`.
    weight = tf * idf  (use index["idf"].get(term, 0.0); unseen terms -> 0).
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


def cosine(u: dict, v: dict) -> float:
    """Cosine similarity between two sparse {term: weight} vectors.

    dot / (||u|| * ||v||). Return 0.0 if either vector has zero norm.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    raise NotImplementedError


def search(index: dict, query: str, k: int = 3) -> list[tuple[int, float]]:
    """Return the top-k (doc_id, score) pairs ranked by cosine similarity.

    Build the query's tf-idf vector, score it against every document's
    tf-idf vector, and return the k highest. Break ties by doc_id (ascending).
    """
    # TODO (STEP 4): implement. Check with: pytest -k step4
    raise NotImplementedError


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
        print("search.py is not implemented yet, fill in the TODOs, then re-run.")
