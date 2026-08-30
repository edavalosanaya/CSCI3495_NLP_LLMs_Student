"""Tests for W11C2 rag (offline pipeline: chunking, retrieval, citations).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-11/class-02/exercise/test_rag.py -k step1 -q

Set RAG_FROM=solution to test the reference solution.
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "rag.py"
    if os.environ.get("RAG_FROM") == "solution"
    else _HERE / "rag.py"
)
_spec = importlib.util.spec_from_file_location("rag_under_test", _SRC)
rag = importlib.util.module_from_spec(_spec)
sys.modules["rag_under_test"] = rag
_spec.loader.exec_module(rag)


def _implemented(fn, *args):
    try:
        fn(*args)
        return True
    except NotImplementedError:
        return False


def test_step1_chunking_splits_on_blank_lines():
    chunks = rag.chunk_documents([("doc1.md", "first passage\n\nsecond passage")])
    assert len(chunks) == 2
    assert chunks[0].text == "first passage"
    assert chunks[1].text == "second passage"
    assert chunks[0].id == 0 and chunks[1].id == 1
    assert all(c.source == "doc1.md" for c in chunks)


def test_step1_chunking_full_corpus_ids_sequential():
    chunks = rag.chunk_documents(rag.NOTES)
    assert len(chunks) >= 5
    assert [c.id for c in chunks] == list(range(len(chunks)))


def _build_retriever():
    chunks = rag.chunk_documents(rag.NOTES)
    return rag.TfidfRetriever(chunks)


def test_step2_retrieve_relevant_chunk_for_cot():
    if not _implemented(rag.TfidfRetriever(rag.chunk_documents(rag.NOTES)).retrieve, "x", 1):
        pytest.skip("retrieve not implemented")
    r = _build_retriever()
    top = r.retrieve("What does chain-of-thought prompting add?", k=2)
    assert len(top) == 2
    # The most relevant chunk should mention chain-of-thought / reasoning.
    assert any("chain-of-thought" in c.text.lower() or "reasoning" in c.text.lower()
               for c in top)


def test_step2_retrieve_relevant_chunk_for_rag():
    if not _implemented(rag.TfidfRetriever(rag.chunk_documents(rag.NOTES)).retrieve, "x", 1):
        pytest.skip("retrieve not implemented")
    r = _build_retriever()
    top = r.retrieve("How does RAG reduce hallucination?", k=1)
    assert "hallucination" in top[0].text.lower()


def test_step3_build_prompt_grounded():
    if not _implemented(rag.build_prompt, "q", []):
        pytest.skip("build_prompt not implemented")
    chunks = rag.chunk_documents(rag.NOTES)[:2]
    p = rag.build_prompt("What is X?", chunks)
    assert "[1]" in p and "[2]" in p          # numbered context
    assert "What is X?" in p                   # question present
    assert "don't know" in p.lower() or "do not know" in p.lower()  # grounding instruction


def test_step4_verify_citations():
    if not _implemented(rag.verify_citations, "x", []):
        pytest.skip("verify_citations not implemented")
    retrieved = rag.chunk_documents(rag.NOTES)[:3]
    assert rag.verify_citations("Answer here. [1] and [3]", retrieved) == {1, 3}
    # [9] is out of range -> hallucinated, must be excluded.
    assert rag.verify_citations("See [2] and [9].", retrieved) == {2}
    assert rag.verify_citations("No citations at all.", retrieved) == set()


def test_step5_stub_generator_offline_endtoend():
    """The whole offline pipeline runs and produces a valid citation."""
    if not all([
        _implemented(rag.build_prompt, "q", []),
        _implemented(rag.verify_citations, "x", []),
        _implemented(rag.TfidfRetriever(rag.chunk_documents(rag.NOTES)).retrieve, "x", 1),
    ]):
        pytest.skip("pipeline functions not implemented")
    chunks = rag.chunk_documents(rag.NOTES)
    r = rag.TfidfRetriever(chunks)
    top = r.retrieve("How does RAG reduce hallucination?", k=3)
    prompt = rag.build_prompt("How does RAG reduce hallucination?", top)
    ans = rag.StubGenerator().answer(prompt)
    assert rag.verify_citations(ans, top) == {1}
