"""Tests for HW6 (ReAct agent with tool use).

Defaults to the student's ``homeworks/hw6/agent.py``. Set HW6_FROM=solution to
test the reference solution (used by the course sweep). Skips gracefully when
unimplemented.

The ReAct loop is driven by a MOCK llm returning canned turns, so NO Ollama is
needed. A separate opt-in live test exercises real Ollama and skips cleanly.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "agent.py"
    if os.environ.get("HW6_FROM") == "solution"
    else _HERE.parent / "agent.py"
)
_spec = importlib.util.spec_from_file_location("hw6_agent_under_test", _SRC)
agent = importlib.util.module_from_spec(_spec)
sys.modules["hw6_agent_under_test"] = agent
_spec.loader.exec_module(agent)


def _implemented() -> bool:
    try:
        agent.calculator("1 + 1")
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="HW6 agent.py not implemented yet (fill in the TODOs)"
)


# --- Task 1: tools ---------------------------------------------------------
@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 3", "5"),
        ("12 * (3 + 4)", "84"),
        ("10 / 4", "2.5"),
        ("-3 + 5", "2"),
        ("2 * -3", "-6"),
    ],
)
def test_step1_calculator_ok(expr, expected):
    assert agent.calculator(expr) == expected


def test_step1_calculator_is_safe_and_errors():
    # division by zero -> Error, not an exception
    assert agent.calculator("1 / 0").startswith("Error")
    # malformed -> Error
    assert agent.calculator("2 +").startswith("Error")
    # must NOT execute arbitrary code (no eval): a name reference is invalid
    assert agent.calculator("__import__('os').system('echo hi')").startswith("Error")


def test_step2_search_finds_and_misses():
    # "france" matches the France entry, which states its capital is Paris.
    assert "Paris" in agent.search("What is the capital of France?")
    assert "Guido" in agent.search("who created python")
    # an explicit Paris query hits the Paris entry
    assert "2.1 million" in agent.search("population of Paris")
    assert agent.search("zxqw nonsense token") == "No results found."


# --- Task 2: parsing -------------------------------------------------------
def test_step3_parse_action_step():
    text = "Thought: I should compute it.\nAction: calculator\nAction Input: 2 + 2"
    step = agent.parse_step(text)
    assert step.final_answer is None
    assert step.action == "calculator"
    assert step.action_input == "2 + 2"
    assert "compute" in step.thought


def test_step3_parse_final_answer_wins():
    text = "Thought: done.\nAction: search\nAction Input: x\nFinal Answer: 42"
    step = agent.parse_step(text)
    assert step.final_answer == "42"


def test_step3_parse_is_case_insensitive_and_trims():
    text = "thought:  reasoning here \naction:  search \naction input:  paris "
    step = agent.parse_step(text)
    assert step.action == "search"
    assert step.action_input == "paris"
    assert step.thought == "reasoning here"


# --- Task 3: dispatch ------------------------------------------------------
def test_step4_run_tool_dispatch():
    s = agent.Step(action="calculator", action_input="6 * 7")
    assert agent.run_tool(s) == "42"
    s2 = agent.Step(action="bogus", action_input="x")
    assert agent.run_tool(s2).startswith("Error: unknown tool")


# --- Task 4: ReAct loop with a MOCK llm ------------------------------------
def make_scripted_llm(turns):
    """Return an llm(prompt)->str that yields the given turns in order."""
    seq = iter(turns)

    def _llm(prompt):
        return next(seq)

    return _llm


def test_step6_react_loop_uses_tool_then_answers():
    turns = [
        "Thought: I need to multiply.\nAction: calculator\nAction Input: 6 * 7",
        "Thought: Now I know.\nFinal Answer: 42",
    ]
    out = agent.react_loop("What is 6 times 7?", make_scripted_llm(turns), max_steps=5)
    assert out["answer"] == "42"
    assert out["steps"] == 2
    # the tool observation was recorded in the transcript
    assert any("Observation: 42" in line for line in out["history"])


def test_step6_react_loop_multi_tool():
    turns = [
        "Thought: look it up.\nAction: search\nAction Input: capital of France",
        "Thought: got it.\nFinal Answer: Paris",
    ]
    out = agent.react_loop("Capital of France?", make_scripted_llm(turns))
    assert out["answer"] == "Paris"
    # the search observation (the France entry, which names Paris) is recorded
    assert any("capital is Paris" in line for line in out["history"])


def test_step6_react_loop_respects_max_steps():
    # Never produces a Final Answer -> stops at max_steps with answer None.
    loop_turn = "Thought: keep going.\nAction: calculator\nAction Input: 1 + 1"
    out = agent.react_loop("loop forever", make_scripted_llm([loop_turn] * 10), max_steps=3)
    assert out["answer"] is None
    assert out["steps"] == 3


def test_step5_build_react_prompt_includes_question_and_tools():
    p = agent.build_react_prompt("Q?", ["Thought: a", "Observation: b"])
    assert "Q?" in p
    assert "calculator" in p and "search" in p
    assert "Observation: b" in p


# --- Optional: live Ollama path, skips cleanly when unavailable ------------
@pytest.mark.skipif(
    os.environ.get("HW6_LIVE_OLLAMA") != "1",
    reason="set HW6_LIVE_OLLAMA=1 to run the live Ollama agent test",
)
def test_step6_react_loop_live_ollama():
    try:
        llm = agent.ollama_llm()
        out = agent.react_loop("What is 21 * 2?", llm, max_steps=5)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Ollama unavailable: {e}")
    assert isinstance(out, dict) and "answer" in out


def test_step3_parse_step_on_unlabelled_text():
    """A model that ignores the format must not crash the loop."""
    s = agent.parse_step("I think the answer might be 42 but I am not sure.")
    assert s.action is None and s.final_answer is None


def test_step4_run_tool_reports_unknown_tools_instead_of_raising():
    out = agent.run_tool(agent.Step(thought=None, action="teleport",
                                    action_input="home", final_answer=None))
    assert isinstance(out, str)
    assert "teleport" in out.lower() or "unknown" in out.lower()
