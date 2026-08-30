#!/usr/bin/env python3
"""W11C2 starter, a minimal RAG pipeline over course notes.

Work through the lab in `README.md`. Each STEP below has its own check
(python -m pytest ... -k step1 -q). Or run everything:
    python -m pytest weeks/week-11/class-02/exercise/test_rag.py -q

End-to-end (real local model if available, else stub generator):
    python weeks/week-11/class-02/exercise/rag.py

Everything except the optional LLM call runs fully offline (TF-IDF retriever).
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass

# Load the notes corpus by absolute path so the script works from any cwd.
import importlib.util as _ilu
from pathlib import Path as _Path

_notes_path = _Path(__file__).resolve().parent / "notes.py"
_notes_spec = _ilu.spec_from_file_location("rag_notes", _notes_path)
_notes_mod = _ilu.module_from_spec(_notes_spec)
_notes_spec.loader.exec_module(_notes_mod)
NOTES = _notes_mod.NOTES

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")


@dataclass
class Chunk:
    id: int
    source: str
    text: str


# ----------------------------- STEP 1 -----------------------------
def chunk_documents(docs: list[tuple[str, str]]) -> list[Chunk]:
    """Split each (source, text) doc into passages on blank lines.

    Return a flat list of Chunk with sequential ids starting at 0. Strip
    whitespace and drop empty passages.
    """
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    chunks: list[Chunk] = []
    cid = 0
    for source, text in docs:
        for passage in re.split(r"\n\s*\n", text):
            passage = passage.strip()
            if passage:
                chunks.append(Chunk(id=cid, source=source, text=passage))
                cid += 1
    return chunks


class TfidfRetriever:
    """Offline retriever: cosine similarity over TF-IDF vectors (scikit-learn)."""

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    # --------------------------- STEP 2 ---------------------------
    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        """Return the top-k chunks most similar to the query.

        Hint: transform the query with self.vectorizer, compute cosine
        similarity against self.matrix (sklearn cosine_similarity), and take the
        k highest-scoring chunks (descending).
        """
        # TODO (STEP 2): implement. Check with: pytest -k step2
        raise NotImplementedError


# ----------------------------- STEP 3 -----------------------------
def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Grounded prompt: numbered context [1..k] + instructions + the question.

    Instruct the model to use ONLY the context, cite sources like [1], and say
    "I don't know." if the answer is not in the context.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    raise NotImplementedError


# ----------------------------- STEP 4 -----------------------------
def verify_citations(answer: str, retrieved: list[Chunk]) -> set[int]:
    """Return the set of VALID cited indices found in `answer`.

    Citations look like [1], [2], ... A citation is valid if its number is in
    1..len(retrieved). Use this to detect hallucinated citations (out of range).
    """
    # GIVEN (STEP 4): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return {n for n in cited if 1 <= n <= len(retrieved)}


# --------------------------- generator backends ---------------------------
class Generator:
    def answer(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class StubGenerator(Generator):
    """Offline generator: returns the first context line + a citation to [1].

    Not intelligent, but it answers *from the retrieved context* and emits a
    valid citation, so the grounding + citation-checking pipeline is testable.
    """

    def answer(self, prompt: str) -> str:
        # Match the first numbered context chunk: a line that STARTS with "[1] ".
        m = re.search(r"^\[1\]\s*(.+)", prompt, flags=re.MULTILINE)
        snippet = m.group(1).strip() if m else "I don't know."
        first_sentence = snippet.split(". ")[0].strip()
        return f"{first_sentence}. [1]"


class OllamaGenerator(Generator):
    def __init__(self, name: str = MODEL):
        import ollama
        self.client = ollama.Client()
        self.name = name

    def answer(self, prompt: str) -> str:
        resp = self.client.chat(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "seed": 0, "num_predict": 200},
        )
        return resp["message"]["content"]


def get_generator() -> Generator:
    try:
        g = OllamaGenerator()
        g.answer("Say hi.")
        print(f"[model] using Ollama model '{MODEL}'")
        return g
    except Exception as e:  # noqa: BLE001
        print(f"[model] Ollama unavailable ({type(e).__name__}); using offline stub generator.")
        return StubGenerator()


def main() -> int:
    chunks = chunk_documents(NOTES)
    retriever = TfidfRetriever(chunks)
    gen = get_generator()

    questions = [
        "What does chain-of-thought prompting add?",
        "How does RAG reduce hallucination?",
        "What does temperature do during decoding?",
    ]
    for q in questions:
        top = retriever.retrieve(q, k=3)
        prompt = build_prompt(q, top)
        ans = gen.answer(prompt)
        cited = verify_citations(ans, top)
        print(f"\nQ: {q}")
        print(f"  retrieved: {[c.source for c in top]}")
        print(f"  answer: {ans.strip()[:160]}")
        print(f"  valid citations: {sorted(cited)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
