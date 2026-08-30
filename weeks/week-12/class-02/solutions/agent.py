"""W12C2 reference solution, a ReAct agent that grows one tool at a time.

The agent logic (prompt building, action parsing, the loop, the guards) is
fully testable WITHOUT an LLM: we inject an `llm` callable that maps the
running transcript to the model's next chunk of text. Tests pass a scripted
fake; the demo passes an Ollama-backed function.

Action grammar, one per step:
    Action: calc[log(3**2 * 16 - 10)]
    Action: weather[san antonio, yesterday]
    Action: finish[7.0 degrees hotter]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tools import TOOLS

LLM = Callable[[str], str]

# `Action: name[` or `Action: name(` ... the closing delimiter is found by
# scanning, not by regex, because expressions contain their own brackets.
_ACTION_OPEN = re.compile(r"Action:\s*([A-Za-z_]+)\s*([\[\(])")

DESCRIPTIONS = {
    "calc": "calc[expression]      Python arithmetic: ** powers, log(), sqrt().",
    "today": "today[]               today's date as YYYY-MM-DD.",
    "weather": "weather[city, day]    high temperature in F. day may be today, yesterday, or YYYY-MM-DD.",
    "search": "search[query]         look up a fact in the local knowledge base.",
}

# One worked example per shape of task. Keep this SHORT: on a 0.5B model a
# longer prompt measurably makes behaviour worse, not better.
EXAMPLES = """Example
Task: What is sqrt(45 + 19)?
Thought: I will pass the whole expression to the calculator.
Action: calc[sqrt(45 + 19)]
Observation: 8.0
Thought: The calculator answered, so I am done.
Action: finish[8.0]
"""

DATE_EXAMPLE = """Example
Task: What is today's date?
Thought: The date is not something I can know, so I will ask the tool.
Action: today[]
Observation: 2026-01-31
Thought: The tool gave the date, so I am done.
Action: finish[2026-01-31]
"""

MULTI_EXAMPLE = """Example
Task: How much warmer was Austin today than yesterday?
Thought: First I need today's temperature.
Action: weather[austin, today]
Observation: 99.0
Thought: Now I need yesterday's temperature.
Action: weather[austin, yesterday]
Observation: 96.0
Thought: Now I subtract those two numbers with the calculator.
Action: calc[99.0 - 96.0]
Observation: 3.0
Thought: I have the difference, so I am done.
Action: finish[3.0 degrees warmer]
"""


def build_prompt(tools: dict, multi_step: bool = False) -> str:
    """Build the system prompt from the tools that are actually registered.

    The prompt grows as the registry grows, which is the point of the lab: the
    model can only reach for a tool it has been told exists.
    """
    lines = [DESCRIPTIONS.get(name, f"{name}[input]") for name in tools]
    lines.append("finish[answer]        give the final answer and stop.")
    return (
        "You answer questions by calling tools. Reply with EXACTLY two lines:\n"
        "Thought: <one short sentence>\n"
        "Action: <tool>[<input>]\n\n"
        "Tools:\n- " + "\n- ".join(lines) + "\n\n"
        "Rules:\n"
        "1. Put the ENTIRE expression into ONE calc call. Never do arithmetic yourself.\n"
        "2. A tool input is plain text. Never nest one tool call inside another.\n"
        "3. If the last Observation already answers the Task, your next Action\n"
        "   MUST be finish[that answer]. Never repeat an Action that worked.\n\n"
        + (MULTI_EXAMPLE if multi_step
           else EXAMPLES + ("\n" + DATE_EXAMPLE if "today" in tools else ""))
        + "\nNow do the same for the new task. Stop after your Action line.\n"
    )


@dataclass
class Step:
    thought: str
    tool: str
    tool_input: str
    observation: str


@dataclass
class Trace:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: Optional[str] = None
    stopped_reason: str = ""   # finished | budget | stuck | ungrounded

    @property
    def succeeded(self) -> bool:
        return self.answer is not None

    def observations(self) -> list[str]:
        return [s.observation for s in self.steps]


def parse_action(text: str) -> Optional[tuple[str, str]]:
    """Extract the FIRST `Action: tool[input]` from model text.

    Scans for the delimiter that matches the one it opened with, so nested
    brackets survive: `calc[log(3**2 * 16 - 10)]` keeps its whole expression.
    Accepts `tool(input)` too, because small models slip into parentheses and
    there is nothing to be gained by failing on that.
    """
    m = _ACTION_OPEN.search(text)
    if not m:
        return None
    tool, opener = m.group(1), m.group(2)
    closer = "]" if opener == "[" else ")"
    depth, out, i = 1, [], m.end()
    while i < len(text):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return tool, "".join(out).strip()
        out.append(ch)
        i += 1
    return None   # never closed: treat as malformed


def parse_thought(text: str) -> str:
    m = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_tool(tool: str, tool_input: str, tools: dict = TOOLS) -> str:
    """Dispatch to a registered tool, turning ALL errors into Observations."""
    fn = tools.get(tool)
    if fn is None:
        return f"Error: unknown tool '{tool}'. Available: {', '.join(tools)}."
    try:
        return fn(tool_input)
    except Exception as e:  # noqa: BLE001, a tool must never crash the loop
        return f"Error: tool '{tool}' failed: {e}"


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def is_grounded(answer: str, observations: list[str]) -> bool:
    """True if every number in the answer also appears in some Observation.

    Why this exists: with tools available, the most common failure of a small
    model is not a broken tool call, it is SKIPPING one and writing a
    plausible number from memory instead. The loop cannot tell a real lookup
    from an invented one, but this check can.
    """
    seen = {n for obs in observations for n in _NUM_RE.findall(obs)}
    return all(n in seen for n in _NUM_RE.findall(answer))


def _fall_back_to_last_observation(trace: Trace) -> None:
    """If the loop ended without a `finish`, answer with the last good Observation.

    Small models very often call the right tool, read the right answer, and
    then simply never say "finish". Throwing that work away would be silly, so
    the LOOP supplies the ending the model failed to write. The trace still
    records why (`stopped_reason`), because a fallback answer is weaker
    evidence than a deliberate one.
    """
    if trace.answer is not None:
        return
    for step in reversed(trace.steps):
        if step.tool and step.tool != "finish" and not step.observation.startswith("Error:"):
            trace.answer = step.observation
            trace.stopped_reason += "+fellback"
            return


def run_agent(task: str, llm: LLM, tools: dict = TOOLS, max_steps: int = 6,
              multi_step: bool = False, require_grounded: bool = False) -> Trace:
    """Run the ReAct loop.

    Guards, each earning its place:
      * `max_steps`      a hard budget, so a confused model cannot loop forever
      * malformed output a corrective Observation instead of a crash
      * repeated action  stop as "stuck" rather than burn the budget
      * tool errors      become Observations the model can read and recover from
      * `require_grounded` reject a final answer containing invented numbers
    """
    trace = Trace(task=task)
    transcript = f"{build_prompt(tools, multi_step)}\nTask: {task}\n"
    last_action: Optional[tuple[str, str]] = None

    for _ in range(max_steps):
        text = llm(transcript)
        thought = parse_thought(text)
        action = parse_action(text)

        if action is None:
            obs = "Error: no valid Action found. Use `Action: tool[input]`."
            transcript += f"Thought: {thought}\nObservation: {obs}\n"
            trace.steps.append(Step(thought, "", "", obs))
            continue

        tool, tool_input = action

        if tool == "finish":
            if require_grounded and not is_grounded(tool_input, trace.observations()):
                trace.stopped_reason = "ungrounded"
                trace.steps.append(Step(thought, "finish", tool_input,
                                        "Error: answer contains a number no tool returned."))
                return trace
            trace.answer = tool_input
            trace.stopped_reason = "finished"
            trace.steps.append(Step(thought, "finish", tool_input, tool_input))
            return trace

        if action == last_action:
            trace.stopped_reason = "stuck"
            trace.steps.append(Step(thought, tool, tool_input,
                                    "Error: repeated action; stopping."))
            _fall_back_to_last_observation(trace)
            return trace
        last_action = action

        obs = run_tool(tool, tool_input, tools)
        trace.steps.append(Step(thought, tool, tool_input, obs))
        transcript += f"Thought: {thought}\nAction: {tool}[{tool_input}]\nObservation: {obs}\n"

    trace.stopped_reason = "budget"
    _fall_back_to_last_observation(trace)
    return trace


# --------------------------------------------------------------------------
# Real-LLM backend (optional): Ollama. Degrades gracefully when absent.
# --------------------------------------------------------------------------
# This lab needs a model that can CHAIN tool calls. Measured on the two small
# models the course uses: qwen2.5:0.5b calls the right tool and reads the right
# number, but almost never emits `finish` and cannot plan a 3-call chain;
# qwen2.5:1.5b does all five lab tasks cleanly. Both are far under the course's
# 3B ceiling and both run on a laptop CPU, so this lab defaults to the 1.5b and
# Step 7 has you run the 0.5b to see where the floor is.
DEFAULT_MODEL = "qwen2.5:1.5b"
COURSE_MODEL = os.environ.get("COURSE_MODEL", DEFAULT_MODEL)


def make_ollama_llm(model: str = COURSE_MODEL) -> LLM:
    """An LLM callable backed by a local Ollama model.

    `stop` at "Observation:" is what keeps the model from writing its own
    observations: the ENVIRONMENT supplies those, never the model.
    """
    import ollama  # imported lazily so the tests never need it

    client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))

    def llm(transcript: str) -> str:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": transcript}],
            options={"temperature": 0.0, "stop": ["Observation:"]},
        )
        return resp["message"]["content"]

    return llm


def ask_directly(question: str, model: str = COURSE_MODEL) -> str:
    """Ask the model with NO tools at all. This is the 'before' picture."""
    import ollama

    client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.0},
    )
    return resp["message"]["content"].strip()


def ollama_available(model: str = COURSE_MODEL) -> bool:
    """True iff a local Ollama server has the model pulled."""
    try:
        import ollama

        client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))
        names = [m.get("model", "") for m in client.list().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:  # noqa: BLE001
        return False
