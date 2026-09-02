"""W12C2 tests. Run one step at a time:  pytest -k step3

Every test here runs WITHOUT an LLM. The agent takes its model as a callable,
so a scripted fake is enough to test the loop, the guards and the parsing.
The real model only shows up in run_demo.py.
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
# The course sweep runs this same file a second time with AGENT_FROM=solution
# to verify the reference implementation.
_SRC = _HERE.parent / "solutions" if os.environ.get("AGENT_FROM") == "solution" else _HERE


def _load_pair(src: Path):
    """Import this lab's `tools` and `agent` from one source directory.

    `agent.py` does `from tools import TOOLS`, so `tools` has to be importable
    under its plain name while `agent` executes. We therefore borrow the plain
    names for the duration and put sys.modules back exactly as we found it,
    because later weeks in the same pytest session have their own `agent.py`
    and must not pick up this one.
    """
    saved = {k: sys.modules.get(k) for k in ("tools", "agent")}
    sys.path.insert(0, str(src))
    try:
        loaded = {}
        for name in ("tools", "agent"):
            spec = importlib.util.spec_from_file_location(name, src / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            loaded[name] = mod
        return loaded["tools"], loaded["agent"]
    finally:
        sys.path.remove(str(src))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# No fallback to the reference solution. It used to load ../solutions when the
# starter still raised NotImplementedError, which meant a student could write
# nothing, run pytest, see everything pass, and believe the lab was finished.
# The course sweep verifies the reference by setting AGENT_FROM=solution
# (scripts/test_all.sh exports every *_FROM var), so nothing needs the fallback.
T, A = _load_pair(_SRC)


# STEP 5 is a dict, not a function, so it cannot raise NotImplementedError and
# the conftest hook cannot turn it into a skip. Gate on it being empty instead,
# so an unwritten registry reports "skipped" like every other unwritten step
# rather than failing with a confusing set-difference.
needs_registry = pytest.mark.skipif(
    not T.TOOLS, reason="not written yet (fill in this step's TODO)"
)


def scripted(*replies: str):
    """A fake LLM that returns the given replies in order, then repeats the last."""
    box = list(replies)

    def llm(_transcript: str) -> str:
        return box.pop(0) if len(box) > 1 else box[0]

    return llm


# --------------------------------------------------------------- step 1: calc
def test_step1_calc_evaluates_a_whole_expression():
    assert T.calculator("log(3^2 * 16 - 10)").startswith("4.8978")


def test_step1_calc_rewrites_caret_as_power():
    # In Python `^` is XOR, so an unrewritten `3^2` would be 1, not 9.
    assert T.calculator("3^2") == "9"


def test_step1_calc_turns_errors_into_observations():
    assert T.calculator("1/0") == "Error: division by zero"
    assert T.calculator("").startswith("Error:")
    assert T.calculator("__import__('os')").startswith("Error:")


# -------------------------------------------------------------- step 2: today
def test_step2_today_returns_the_real_date():
    assert T.today() == datetime.date.today().isoformat()


# ------------------------------------------------------------ step 3: weather
def test_step3_weather_reads_today_and_yesterday():
    assert T.weather("san antonio, today") == "101.0"
    assert T.weather("san antonio, yesterday") == "94.0"


def test_step3_weather_forgives_sloppy_input():
    assert T.weather("San_Antonio, TODAY") == "101.0"
    assert T.weather('austin, "yesterday"') == "96.0"


def test_step3_weather_accepts_an_iso_date():
    y = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
    assert T.weather(f"austin, {y}") == "95.0"


def test_step3_weather_errors_are_readable():
    assert T.weather("paris, today").startswith("Error: no weather for 'paris'")
    assert T.weather("austin, 1999-01-01").startswith("Error: no reading that far back")


# ------------------------------------------------------------- step 4: search
def test_step4_search_finds_the_right_entry():
    assert "Shinn" in T.search("reflexion")
    assert "Paris" in T.search("what is the capital of france")


# ----------------------------------------------------------- step 5: registry
@needs_registry
def test_step5_registry_has_all_four_tools():
    assert set(T.TOOLS) == {"calc", "today", "weather", "search"}


@needs_registry
def test_step5_prompt_lists_every_registered_tool():
    prompt = A.build_prompt(T.TOOLS)
    for name in T.TOOLS:
        assert name in prompt
    assert "finish" in prompt


@needs_registry
def test_step5_unknown_tool_becomes_an_observation():
    out = A.run_tool("teleport", "mars", T.TOOLS)
    assert out.startswith("Error: unknown tool 'teleport'")


# ------------------------------------------------------------ step 6: parsing
def test_step6_parse_keeps_nested_brackets():
    assert A.parse_action("Action: calc[log(3**2 * 16 - 10)]") == (
        "calc", "log(3**2 * 16 - 10)")


def test_step6_parse_accepts_parentheses():
    assert A.parse_action("Action: finish(7 degrees)") == ("finish", "7 degrees")


def test_step6_parse_returns_none_when_malformed():
    assert A.parse_action("I think the answer is 42.") is None
    assert A.parse_action("Action: calc[1 + 2") is None


# ---------------------------------------------------------- step 7: grounding
def test_step7_grounded_accepts_numbers_that_came_from_tools():
    assert A.is_grounded("7 degrees hotter", ["101.0", "94.0", "7"])


def test_step7_grounded_rejects_an_invented_number():
    # The classic failure: the model looked up today, invented yesterday.
    assert not A.is_grounded("3 degrees hotter", ["101.0"])
    assert not A.is_grounded("98.0", ["101.0", "94.0"])


# ------------------------------------------------------- the loop and guards
def test_loop_finishes_and_reports_the_answer():
    llm = scripted("Thought: compute it.\nAction: calc[2 + 2]",
                   "Thought: done.\nAction: finish[4]")
    tr = A.run_agent("2+2?", llm, T.TOOLS)
    assert tr.answer == "4" and tr.stopped_reason == "finished"


def test_loop_stops_on_a_repeated_action():
    llm = scripted("Thought: again.\nAction: calc[2 + 2]")
    tr = A.run_agent("2+2?", llm, T.TOOLS)
    assert "stuck" in tr.stopped_reason


def test_loop_falls_back_to_the_last_good_observation():
    # A small model often calls the right tool and then never says finish.
    llm = scripted("Thought: compute.\nAction: calc[2 + 2]")
    tr = A.run_agent("2+2?", llm, T.TOOLS)
    assert tr.answer == "4" and "fellback" in tr.stopped_reason


def test_loop_survives_malformed_output():
    llm = scripted("I have no idea what to do.",
                   "Thought: ok.\nAction: finish[42]")
    tr = A.run_agent("?", llm, T.TOOLS)
    assert tr.answer == "42"
    assert tr.steps[0].observation.startswith("Error: no valid Action")


def test_loop_respects_the_step_budget():
    llm = scripted("Thought: a.\nAction: calc[1+1]", "Thought: b.\nAction: calc[2+2]",
                   "Thought: c.\nAction: calc[3+3]", "Thought: d.\nAction: calc[4+4]")
    tr = A.run_agent("?", llm, T.TOOLS, max_steps=3)
    assert "budget" in tr.stopped_reason and len(tr.steps) == 3


def test_loop_rejects_an_ungrounded_answer():
    llm = scripted("Thought: look it up.\nAction: weather[san antonio, today]",
                   "Thought: done.\nAction: finish[3 degrees hotter]")
    tr = A.run_agent("?", llm, T.TOOLS, require_grounded=True)
    assert tr.stopped_reason == "ungrounded" and tr.answer is None
