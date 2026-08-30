"""W13C1 tests, memory, planning, and Reflexion, all with a MOCK LLM.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-13/class-01/exercise/test_agent.py -k step1 -q

Default: run against the student exercise files. To check the reference
solution (used by the course sweep): set  AGENT_FROM=solution
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "solutions" if os.environ.get("AGENT_FROM") == "solution" else _HERE


def _load(name: str):
    path = _SRC / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"w13c1_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SRC))
    sys.modules[f"w13c1_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


tools = _load("tools")
agent = _load("agent")
scripted = _load("scripted_llm")
ScriptedLLM = scripted.ScriptedLLM
ReflexiveLLM = scripted.ReflexiveLLM
const = scripted.const


# ----------------------------- memory -----------------------------
def test_step1_memory_starts_empty():
    assert agent.Memory().as_prompt() == ""


def test_step1_memory_renders_notes():
    m = agent.Memory()
    m.add("search first")
    m.add("then compute")
    p = m.as_prompt()
    assert "search first" in p and "then compute" in p


# ----------------------------- planner -----------------------------
def test_step3_plan_injected_into_prompt():
    # With a planner, the plan text should appear in the attempt's prompt.
    seen = {}

    def capture_llm(transcript):
        seen["t"] = transcript
        return "Thought: ok.\nAction: finish[done]"

    agent.react_attempt(
        "do a thing",
        capture_llm,
        agent.Memory(),
        plan="1. step one\n2. step two",
    )
    assert "step one" in seen["t"]


def test_step2_make_plan_none_planner_is_empty():
    assert agent.make_plan("task", None) == ""


# ------------------------- single attempt -------------------------
def test_step3_attempt_uses_memory_in_prompt():
    mem = agent.Memory()
    mem.add("LESSON-XYZ")
    seen = {}

    def capture(transcript):
        seen["t"] = transcript
        return "Thought: ok.\nAction: finish[ok]"

    agent.react_attempt("t", capture, mem)
    assert "LESSON-XYZ" in seen["t"]


# --------------------------- reflexion ----------------------------
def test_step4_reflexion_succeeds_first_try_no_extra_attempts():
    llm = ScriptedLLM([
        "Thought: easy.\nAction: calc[2*3]",
        "Thought: done.\nAction: finish[6]",
    ])
    trace, mem, attempts = agent.run_reflexion_agent(
        "2*3?", llm, success_check=lambda a: a == "6", max_attempts=3
    )
    assert attempts == 1
    assert trace.succeeded and not mem.notes  # no reflection needed


def test_step4_reflexion_recovers_after_failure():
    # Fails attempt 1, reflects, then succeeds, the core Reflexion behavior.
    before = ["Thought: guess.\nAction: finish[wrong]"]
    after = [
        "Thought: search now.\nAction: search[paris]",
        "Thought: double it.\nAction: calc[2100000 * 2]",
        "Thought: done.\nAction: finish[4200000]",
    ]
    llm = ReflexiveLLM(before, after)
    trace, mem, attempts = agent.run_reflexion_agent(
        "population of capital of France doubled?",
        llm,
        success_check=lambda a: "4200000" in a,
        max_attempts=3,
    )
    assert attempts == 2
    assert trace.succeeded
    assert len(mem.notes) == 1  # exactly one reflection was stored


def test_step4_reflexion_gives_up_after_max_attempts():
    llm = ScriptedLLM(["Thought: nope.\nAction: finish[wrong]"] * 20)
    trace, mem, attempts = agent.run_reflexion_agent(
        "impossible", llm, success_check=lambda a: a == "right", max_attempts=3
    )
    assert attempts == 3
    assert not (trace.succeeded and trace.answer == "right")
    assert len(mem.notes) == 3  # reflected after each failure


def test_step4_reflector_callable_is_used():
    llm = ScriptedLLM(["Thought: x.\nAction: finish[bad]"] * 10)
    _, mem, _ = agent.run_reflexion_agent(
        "t",
        llm,
        success_check=lambda a: False,
        reflector=const("CUSTOM-REFLECTION"),
        max_attempts=2,
    )
    assert any("CUSTOM-REFLECTION" in n for n in mem.notes)
