"""Tests for HW5 (Prompt Engineering & RAG).

Defaults to the student's ``homeworks/hw5/rag.py``. Set HW5_FROM=solution to
test the reference solution (used by the course sweep). Skips gracefully when
the student has not implemented the module yet.

All retriever / chunking / prompt tests run OFFLINE. The end-to-end test uses an
injected MOCK generator, so it never needs Ollama. A separate, opt-in live test
calls real Ollama and SKIPS cleanly when it is unavailable.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "rag.py"
    if os.environ.get("HW5_FROM") == "solution"
    else _HERE.parent / "rag.py"
)
_spec = importlib.util.spec_from_file_location("hw5_rag_under_test", _SRC)
rag = importlib.util.module_from_spec(_spec)
sys.modules["hw5_rag_under_test"] = rag
_spec.loader.exec_module(rag)


def _implemented() -> bool:
    try:
        rag.tokenize("hello world")
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="HW5 rag.py not implemented yet (fill in the TODOs)"
)

CORPUS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Photosynthesis lets plants convert sunlight into chemical energy.",
    "Paris is the capital of France and sits on the river Seine.",
    "The mitochondrion is the powerhouse of the cell.",
    "France uses the euro as its official currency.",
]


def make_index():
    return rag.TfidfIndex().build(CORPUS)


# --- Task 1: tokenize + TF-IDF retriever -----------------------------------
def test_step1_tokenize():
    assert rag.tokenize("Hello, WORLD! 42") == ["hello", "world", "42"]


def test_step1_cosine_basics():
    assert rag.cosine({"a": 1.0}, {}) == 0.0
    assert rag.cosine({"a": 1.0, "b": 0.0}, {"a": 2.0}) == pytest.approx(1.0)
    # orthogonal
    assert rag.cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_step2_index_builds_idf_and_vectors():
    idx = make_index()
    assert len(idx.vectors) == len(CORPUS)
    # "paris" appears in 2 of 5 docs => idf = ln(6/3)+1
    import math

    assert idx.idf["paris"] == pytest.approx(math.log(6 / 3) + 1.0)


def test_step3_search_returns_relevant_doc():
    idx = make_index()
    hits = idx.search("What is the capital of France?", k=3)
    assert hits, "expected at least one hit"
    top_doc = hits[0][0]
    assert "capital of France" in idx.docs[top_doc]
    # scores sorted descending
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_step3_search_irrelevant_query_low_or_empty():
    idx = make_index()
    hits = idx.search("xylophone zebra quantum", k=3)
    assert hits == []  # no overlapping terms -> nothing retrieved


# --- Task 2: chunking ------------------------------------------------------
def test_step4_chunk_overlap_and_coverage():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = rag.chunk_text(words, max_words=10, overlap=4)
    # step = 6: starts at 0,6,12,18,24
    assert chunks[0].split()[:3] == ["w0", "w1", "w2"]
    assert chunks[1].split()[0] == "w6"  # advanced by step=6
    # overlap: last 4 words of chunk0 == first 4 of chunk1
    assert chunks[0].split()[-4:] == chunks[1].split()[:4]
    # every word is covered
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == {f"w{i}" for i in range(25)}


def test_step4_chunk_empty_and_validation():
    assert rag.chunk_text("", max_words=10, overlap=2) == []
    with pytest.raises(ValueError):
        rag.chunk_text("a b c", max_words=5, overlap=5)


# --- Task 3: grounded prompt assembly --------------------------------------
def test_step5_prompt_is_grounded_and_numbered():
    prompt = rag.build_prompt("Where is the Eiffel Tower?", [CORPUS[0], CORPUS[2]])
    low = prompt.lower()
    assert "only" in low and "i don't know" in low  # grounding instruction
    assert "[1]" in prompt and "[2]" in prompt  # numbered citations
    assert "Where is the Eiffel Tower?" in prompt  # the question
    assert "step by step" not in low  # cot off by default


def test_step5_prompt_cot_adds_reasoning_cue():
    prompt = rag.build_prompt("Why?", [CORPUS[1]], cot=True)
    assert "step by step" in prompt.lower()


# --- Task 4: end-to-end with a MOCK generator (no Ollama) ------------------
def test_step6_rag_answer_pipeline_with_mock():
    idx = make_index()
    captured = {}

    def fake_generate(prompt, model="x"):
        captured["prompt"] = prompt
        captured["model"] = model
        return "Paris [3]"

    out = rag.rag_answer(
        idx, "capital of France", k=2, cot=False, model="tiny", generate_fn=fake_generate
    )
    assert out["answer"] == "Paris [3]"
    assert len(out["passages"]) == len(out["retrieved"]) <= 2
    # the generator received the assembled grounded prompt + the chosen model
    assert "capital of France" in captured["prompt"]
    assert captured["model"] == "tiny"
    # retrieved passages are wired into the prompt
    for p in out["passages"]:
        assert p in out["prompt"]


# --- Optional: live Ollama path, skips cleanly when unavailable ------------
@pytest.mark.skipif(
    os.environ.get("HW5_LIVE_OLLAMA") != "1",
    reason="set HW5_LIVE_OLLAMA=1 to run the live Ollama generation test",
)
def test_step6_rag_answer_live_ollama():
    idx = make_index()
    try:
        out = rag.rag_answer(idx, "What is the capital of France?", k=2)
    except Exception as e:  # noqa: BLE001 - Ollama not running / model missing
        pytest.skip(f"Ollama unavailable: {e}")
    assert isinstance(out["answer"], str) and out["answer"]


def test_step5_build_prompt_with_no_passages_still_asks_the_question():
    """Retrieval can come back empty; the prompt must not crash or lose the question."""
    p = rag.build_prompt("what is BPE?", [])
    assert "what is BPE?" in p
    assert isinstance(p, str) and p.strip()


def test_step3_search_respects_k():
    idx = rag.TfidfIndex().build(
        ["alpha beta", "beta gamma", "gamma delta", "delta epsilon"])
    assert len(idx.search("beta", k=1)) <= 1
    assert len(idx.search("beta", k=10)) <= 4, "cannot return more docs than exist"
