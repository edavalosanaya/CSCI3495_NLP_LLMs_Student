#!/usr/bin/env python3
"""W11C2 reference solution, minimal RAG pipeline over course notes."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass

# Load the notes corpus (shared with the exercise) by absolute path.
import importlib.util as _ilu
from pathlib import Path as _Path

_notes_path = _Path(__file__).resolve().parents[1] / "exercise" / "notes.py"
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


def chunk_documents(docs: list[tuple[str, str]]) -> list[Chunk]:
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
    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        # transform, NOT fit_transform: the vectorizer was already fitted on
        # the corpus, and refitting would put the query in its own space.
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix)[0]

        # argsort is ascending, so the BEST scores are at the end. Reverse it,
        # then take the first k.
        order = scores.argsort()[::-1][:k]

        best = []
        for i in order:
            best.append(self.chunks[i])
        return best


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    # Number the chunks from 1 so the model has something to cite.
    numbered = []
    for i, c in enumerate(chunks, start=1):
        numbered.append(f"[{i}] {c.text}")
    context = "\n".join(numbered)
    return (
        "Use ONLY the context below to answer the question. Cite the sources you "
        'use like [1], [2]. If the answer is not in the context, say "I don\'t know."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )


def verify_citations(answer: str, retrieved: list[Chunk]) -> set[int]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return {n for n in cited if 1 <= n <= len(retrieved)}


class Generator:
    def answer(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class StubGenerator(Generator):
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
