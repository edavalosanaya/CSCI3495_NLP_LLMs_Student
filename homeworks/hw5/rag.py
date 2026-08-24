"""HW5 starter, Prompt Engineering & Retrieval-Augmented Generation (RAG).

You will build a small, offline-testable RAG pipeline: index a corpus, retrieve
relevant passages for a query, assemble a grounded (and chain-of-thought)
prompt, and finally generate an answer with a local Ollama model.

References:
  * Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks" (2020), arXiv:2005.11401.
  * Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (2022),
    arXiv:2201.11903.

The retriever and prompt-assembly steps are pure functions you can unit-test
WITHOUT a model. The final generation step calls Ollama and degrades gracefully
when Ollama is not running.

Run the tests with:
    docker compose -f docker/docker-compose.yml run --rm course \
        python -m pytest homeworks/hw5/tests -q
"""
# Each TODO below names its README step. Check one step with:
#     python -m pytest homeworks/hw5/tests -q -k step3      (or step1, step2, ...)
# and the whole assignment with:
#     python -m pytest homeworks/hw5/tests -q

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Step 1, Tokenization & a TF-IDF vector-space retriever (offline).
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric word tokens (regex \\w+)."""
    # TODO (STEP 1): implement
    raise NotImplementedError


@dataclass
class TfidfIndex:
    """A tiny TF-IDF index over a list of documents (passages).

    Attributes you must populate in ``build``:
      docs:  the original passage strings.
      idf:   dict term -> inverse document frequency.
      vectors: list of dicts (term -> tf-idf weight), one per document.
    """

    docs: list[str] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)
    vectors: list[dict[str, float]] = field(default_factory=list)

    def build(self, docs: list[str]) -> "TfidfIndex":
        """Compute idf and per-document tf-idf vectors.

        Use smoothed idf:  idf(t) = ln((1 + N) / (1 + df(t))) + 1
        Term frequency is the raw count of the term in the document.
        A document vector maps each of its terms to ``tf * idf(term)``.
        Returns self.
        """
        # TODO (STEP 2): populate self.docs, self.idf, self.vectors
        raise NotImplementedError

    def _vectorize_query(self, query: str) -> dict[str, float]:
        """Turn a query into a tf-idf vector using the index's idf (terms not
        in the vocabulary get idf 0 / are ignored)."""
        # TODO (STEP 3): implement
        raise NotImplementedError

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Return the top-``k`` (doc_index, cosine_similarity) pairs, sorted by
        similarity descending. Ties broken by lower doc_index. Documents with
        zero similarity may be omitted. Cosine of two zero vectors is 0.0.
        """
        # TODO (STEP 3): implement cosine similarity ranking
        raise NotImplementedError


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse term->weight vectors. 0.0 if either
    has zero norm."""
    # TODO (STEP 1): implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2, Chunking a long document into passages.
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_words: int = 40, overlap: int = 10) -> list[str]:
    """Split ``text`` into overlapping word-windows.

    Each chunk has at most ``max_words`` words; consecutive chunks overlap by
    ``overlap`` words (so the window advances by ``max_words - overlap``).
    The final partial chunk is included. Requires 0 <= overlap < max_words.
    Returns a list of chunk strings (words joined by single spaces).
    """
    # TODO (STEP 4): implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 3, Grounded prompt assembly (with optional chain-of-thought).
# ---------------------------------------------------------------------------
def build_prompt(query: str, passages: list[str], cot: bool = False) -> str:
    """Assemble a grounded RAG prompt.

    The prompt MUST:
      * include a system instruction telling the model to answer ONLY from the
        provided context and to say "I don't know" if the answer is not present;
      * list the retrieved passages, each numbered like "[1] ...", "[2] ...";
      * include the user's question.
    If ``cot`` is True, also instruct the model to reason step by step before
    giving a final answer (chain-of-thought, Wei et al. 2022).

    Return the full prompt as a single string. (Exact wording is up to you, but
    the tests check for the grounding instruction, the numbered citations, the
    question text, and the step-by-step cue when cot=True.)
    """
    # TODO (STEP 5): implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 4, End-to-end RAG (retrieve -> prompt -> generate).
# ---------------------------------------------------------------------------
def generate(prompt: str, model: str = "qwen2.5:0.5b") -> str:
    """Call a local Ollama model and return its text response.

    Use the ``ollama`` Python client. This is the ONLY part that needs a running
    Ollama server; the rest of the pipeline is offline.
    """
    import ollama

    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def rag_answer(
    index: TfidfIndex,
    query: str,
    k: int = 3,
    cot: bool = False,
    model: str = "qwen2.5:0.5b",
    generate_fn=generate,
) -> dict:
    """Full pipeline: retrieve top-k passages, build a grounded prompt, generate.

    Returns a dict with keys:
      'retrieved': list of (doc_index, score) from the retriever,
      'passages':  the retrieved passage strings,
      'prompt':    the assembled prompt,
      'answer':    the generated text.

    ``generate_fn`` is injected so tests can pass a mock (the default calls the
    real Ollama ``generate``).
    """
    # TODO (STEP 6): implement, calling generate_fn(prompt, model=model)
    raise NotImplementedError
