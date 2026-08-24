"""Tests for W14C1 router+workers workflow.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-14/class-01/exercise/test_workflow.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  WORKFLOW_FROM=solution  (used by the course test sweep).

All tests use a deterministic MOCK LLM, no Ollama, no network, so they are
fast and reproducible. This is exactly how you unit-test orchestration logic:
inject canned responses and assert on the routing/dispatch behavior.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "workflow.py"
    if os.environ.get("WORKFLOW_FROM") == "solution"
    else _HERE / "workflow.py"
)
_spec = importlib.util.spec_from_file_location("workflow_under_test", _SRC)
wf = importlib.util.module_from_spec(_spec)
sys.modules["workflow_under_test"] = wf
_spec.loader.exec_module(wf)


# Skip the whole module gracefully if the student hasn't implemented yet.
def _implemented():
    try:
        wf.route("hi", lambda p: "summarize")
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(),
    reason="workflow not implemented yet (fill in the TODOs)",
)


class MockLLM:
    """A canned, scriptable LLM. Returns `router_reply` for routing prompts and
    `worker_reply` for everything else; records every prompt it sees."""

    def __init__(self, router_reply="summarize", worker_reply="[worker output]"):
        self.router_reply = router_reply
        self.worker_reply = worker_reply
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # Heuristic: the router prompt mentions the allowed label set.
        if "summarize, translate, extract" in prompt or "Label:" in prompt:
            return self.router_reply
        return self.worker_reply


# ---- route() ----

@pytest.mark.parametrize("reply,expected", [
    ("summarize", "summarize"),
    ("translate", "translate"),
    ("extract", "extract"),
])
def test_step1_route_valid_labels(reply, expected):
    assert wf.route("anything", MockLLM(router_reply=reply)) == expected


def test_step1_route_is_robust_to_messy_output():
    # Extra words / punctuation / casing must still normalize correctly.
    assert wf.route("x", MockLLM(router_reply="  Summarize.\n")) == "summarize"
    assert wf.route("x", MockLLM(router_reply="TRANSLATE this please")) == "translate"


def test_step1_route_unknown_for_garbage():
    assert wf.route("x", MockLLM(router_reply="banana")) == "unknown"
    assert wf.route("x", MockLLM(router_reply="")) == "unknown"


# ---- workers ----

def test_step2_workers_call_the_llm_and_return_text():
    llm = MockLLM(worker_reply="ok")
    assert wf.worker_summarize("text", llm) == "ok"
    assert wf.worker_translate("text", llm) == "ok"
    assert wf.worker_extract("text", llm) == "ok"


def test_step3_fallback_never_crashes_and_does_not_need_llm():
    # Fallback must be safe even with an LLM that would explode.
    def boom(_):
        raise AssertionError("fallback should not call the LLM")

    msg = wf.worker_fallback("anything", boom)
    assert isinstance(msg, str) and len(msg) > 0


# ---- run_workflow() (the orchestrator) ----

def test_step4_run_workflow_dispatches_to_correct_worker():
    llm = MockLLM(router_reply="translate", worker_reply="bonjour")
    r = wf.run_workflow("Translate hello", llm)
    assert r.label == "translate"
    assert r.output == "bonjour"
    assert "translate" in r.handled_by


def test_step4_run_workflow_routes_unknown_to_fallback():
    llm = MockLLM(router_reply="nonsense")
    r = wf.run_workflow("???", llm)
    assert r.label == "unknown"
    assert "fallback" in r.handled_by
    assert isinstance(r.output, str) and r.output


def test_step4_run_workflow_returns_a_trace():
    llm = MockLLM(router_reply="summarize", worker_reply="a summary")
    r = wf.run_workflow("Summarize this", llm)
    assert isinstance(r.trace, list) and len(r.trace) >= 1
    assert any("route" in step for step in r.trace)


def test_step5_end_to_end_all_paths():
    cases = {
        "summarize": "summarize",
        "translate": "translate",
        "extract": "extract",
        "garbage": "unknown",
    }
    for router_reply, expected_label in cases.items():
        r = wf.run_workflow("query", MockLLM(router_reply=router_reply))
        assert r.label == expected_label
