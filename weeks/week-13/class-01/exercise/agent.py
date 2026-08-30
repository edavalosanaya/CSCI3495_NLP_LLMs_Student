"""W13C1 starter, add MEMORY, PLANNING, and REFLECTION to your agent.

You already have a robust ReAct loop (W12). This week you add three pieces:
  * Memory  , long-term notes (reflections) that persist across attempts.
  * Planner , one up-front step that decomposes the task.
  * Reflexion (Shinn et al., 2023), on failure, write a self-critique to memory
               and RETRY with the lesson in context.

Everything stays testable WITHOUT an LLM: we inject `llm` (and optional
`planner` / `reflector`) callables. Fill in the TODOs.

Run the tests:
    python -m pytest weeks/week-13/class-01/exercise/test_agent.py -q
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tools import TOOLS

LLM = Callable[[str], str]

ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\[(.*?)\]", re.DOTALL)


# --------------------------------------------------------------------------
# Memory  (STEP 1)
# --------------------------------------------------------------------------
@dataclass
class Memory:
    """Long-term notes (e.g. reflections) that persist across attempts."""

    notes: list[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        # GIVEN (STEP 1): written for you. Read it, run its check, and use
        # it as the pattern for the steps you do write.
        self.notes.append(note)

    def as_prompt(self) -> str:
        # GIVEN (STEP 1): written for you. Read it, run its check, and use
        # it as the pattern for the steps you do write.
        if not self.notes:
            return ""
        lines = "\n".join(f"- {n}" for n in self.notes)
        return f"Lessons from previous attempts:\n{lines}\n"


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
    stopped_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.answer is not None


# ----- provided (from W12): parsing + dispatch -----
def parse_action(text: str) -> Optional[tuple[str, str]]:
    m = ACTION_RE.search(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_thought(text: str) -> str:
    m = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_tool(tool: str, tool_input: str) -> str:
    fn = TOOLS.get(tool)
    if fn is None:
        return f"Error: unknown tool '{tool}'. Available: {', '.join(TOOLS)}."
    try:
        return fn(tool_input)
    except Exception as e:  # noqa: BLE001
        return f"Error: tool '{tool}' failed: {e}"


# --------------------------------------------------------------------------
# Planner  (STEP 2)
# --------------------------------------------------------------------------
def make_plan(task: str, planner: Optional[LLM]) -> str:
    """Ask the planner for a short plan. No planner -> empty plan ("")."""
    # TODO (STEP 2): implement. Check with: pytest -k step2
    # and return the stripped result.
    raise NotImplementedError


# --------------------------------------------------------------------------
# Single ReAct attempt (memory + plan injected into the prompt)
# --------------------------------------------------------------------------
def _build_header(task: str, plan: str, memory: Memory) -> str:
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    header = (
        "You solve problems by calling tools. Reply with EXACTLY two lines:\n"
        "Thought: <one short sentence>\n"
        "Action: <tool>[<input>]\n\n"
        "Tools:\n"
        "- calc[expression]   arithmetic, e.g. calc[23 * 4]. Use ** for powers, sqrt().\n"
        "- search[query]      look up a fact you do not know.\n"
        "- finish[answer]     give the final NUMBER and stop.\n\n"
        "Rules: never do arithmetic yourself, always use calc. Put the whole\n"
        "expression in one calc call. finish with the number only.\n\n"
        "Example\n"
        "Task: A shelf has 12 boxes with 6 pens each. How many pens?\n"
        "Thought: I will multiply with the calculator.\n"
        "Action: calc[12 * 6]\n"
        "Observation: 72\n"
        "Thought: The calculator gave the answer.\n"
        "Action: finish[72]\n"
    )
    if plan:
        header += f"\nPlan:\n{plan}\n"
    header += "\n" + memory.as_prompt()
    header += f"\nTask: {task}\n"
    return header


def react_attempt(task: str, llm: LLM, memory: Memory, plan: str = "", max_steps: int = 6) -> Trace:
    """One ReAct attempt (your W12 loop), now with plan + memory in the prompt."""
    trace = Trace(task=task)
    transcript = _build_header(task, plan, memory)
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
            trace.answer = tool_input
            trace.stopped_reason = "finished"
            trace.steps.append(Step(thought, "finish", tool_input, tool_input))
            return trace

        if action == last_action:
            trace.stopped_reason = "stuck"
            trace.steps.append(Step(thought, tool, tool_input, "Error: repeated action; stopping."))
            return trace
        last_action = action

        obs = run_tool(tool, tool_input)
        trace.steps.append(Step(thought, tool, tool_input, obs))
        transcript += f"Thought: {thought}\nAction: {tool}[{tool_input}]\nObservation: {obs}\n"

    trace.stopped_reason = "budget"
    return trace


# --------------------------------------------------------------------------
# Reflection  (STEP 4) , Reflexion (Shinn et al., 2023)
# --------------------------------------------------------------------------
def reflect(task: str, trace: Trace, reflector: Optional[LLM]) -> str:
    """Produce a short verbal self-critique from a FAILED trace."""
    if reflector is None:
        # Deterministic fallback so the loop works with no LLM at all.
        actions = ", ".join(f"{s.tool}[{s.tool_input}]" for s in trace.steps if s.tool)
        return (
            f"Last attempt failed ({trace.stopped_reason}). "
            f"Tried: {actions or 'nothing useful'}. "
            "Next time, search for the key fact first, then compute, then finish."
        )
    # TODO (STEP 4): implement. Check with: pytest -k step4
    # one-or-two-sentence critique; return it stripped.
    raise NotImplementedError


# --------------------------------------------------------------------------
# Reflexion loop  (TODO):  plan -> attempt -> (reflect -> retry)*
# --------------------------------------------------------------------------
def run_reflexion_agent(
    task: str,
    llm: LLM,
    *,
    success_check: Callable[[str], bool],
    planner: Optional[LLM] = None,
    reflector: Optional[LLM] = None,
    max_attempts: int = 3,
    max_steps: int = 6,
) -> tuple[Trace, Memory, int]:
    """Run up to `max_attempts` attempts, reflecting between failures.

    Returns (last_trace, memory, attempts_used). The agent never sees
    `success_check`, it is the evaluator's oracle.
    """
    memory = Memory()
    plan = make_plan(task, planner)
    trace = Trace(task=task)

    # TODO: for attempt in 1..max_attempts:
    #   - trace = react_attempt(task, llm, memory, plan=plan, max_steps=max_steps)
    #   - if trace.succeeded and success_check(trace.answer): return (trace, memory, attempt)
    #   - else: memory.add(reflect(task, trace, reflector))
    # After the loop, return (trace, memory, max_attempts).
    raise NotImplementedError


# ----- optional Ollama backend (provided; degrades gracefully) -----
def make_ollama_llm(model: str = "qwen2.5:0.5b") -> LLM:
    import ollama

    client = ollama.Client()

    def llm(transcript: str) -> str:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": transcript}],
            options={"temperature": 0.0, "stop": ["Observation:"]},
        )
        return resp["message"]["content"]

    return llm


def ollama_available(model: str = "qwen2.5:0.5b") -> bool:
    try:
        import ollama

        names = [m.get("model", "") for m in ollama.Client().list().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:  # noqa: BLE001
        return False
