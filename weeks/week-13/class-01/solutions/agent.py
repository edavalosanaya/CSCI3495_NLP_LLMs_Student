"""W13 reference solution, a reasoning agent with memory, planning, reflection.

We extend the W12 ReAct loop with three architectural pieces (see the
agent-architecture visual):

  * Memory  , short-term (the running trace) + long-term (notes that persist
               across attempts, e.g. reflections).
  * Planner , a single up-front planning step that decomposes the task.
  * Reflection (Reflexion, Shinn et al., 2023), on a failed attempt, write a
               verbal self-critique to memory and RETRY with the lesson in context.

As in W12, everything is testable WITHOUT an LLM: inject an `llm` callable that
maps a transcript to the next chunk of model text. A planner LLM is injected
separately so tests stay deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tools import TOOLS

LLM = Callable[[str], str]

ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\[(.*?)\]", re.DOTALL)


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
@dataclass
class Memory:
    """Short-term trace lives in the transcript; long-term notes persist here."""

    notes: list[str] = field(default_factory=list)  # e.g. reflections

    def add(self, note: str) -> None:
        self.notes.append(note)

    def as_prompt(self) -> str:
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
# Planner
# --------------------------------------------------------------------------
def make_plan(task: str, planner: Optional[LLM]) -> str:
    """Ask the planner LLM for a short plan. No planner -> empty plan."""
    if planner is None:
        return ""
    prompt = (
        "Break the task into 2-4 short numbered steps. Output ONLY the steps.\n"
        f"Task: {task}\nPlan:"
    )
    return planner(prompt).strip()


# --------------------------------------------------------------------------
# Single ReAct attempt (memory + plan injected into the prompt)
# --------------------------------------------------------------------------
def _build_header(task: str, plan: str, memory: Memory) -> str:
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
    """One ReAct attempt, with the plan and long-term memory in the prompt."""
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
# Reflection (Reflexion, Shinn et al., 2023)
# --------------------------------------------------------------------------
def reflect(task: str, trace: Trace, reflector: Optional[LLM]) -> str:
    """Produce a short verbal self-critique from a failed trace."""
    if reflector is None:
        # Deterministic fallback reflection so the loop works without any LLM.
        actions = ", ".join(f"{s.tool}[{s.tool_input}]" for s in trace.steps if s.tool)
        return (
            f"Last attempt failed ({trace.stopped_reason}). "
            f"Tried: {actions or 'nothing useful'}. "
            "Next time, search for the key fact first, then compute, then finish."
        )
    summary = "; ".join(f"{s.tool}[{s.tool_input}] -> {s.observation}" for s in trace.steps if s.tool)
    prompt = (
        "You are reflecting on a FAILED attempt. In one or two sentences, say what "
        "went wrong and what to do differently next time.\n"
        f"Task: {task}\nAttempt: {summary}\nReflection:"
    )
    return reflector(prompt).strip()


# --------------------------------------------------------------------------
# Top-level: plan -> attempt -> (reflect -> retry)*  (Reflexion loop)
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
    """Run up to `max_attempts` ReAct attempts, reflecting between failures.

    `success_check(answer)` decides if the final answer is correct (the agent
    cannot see this; it's the evaluator's oracle). Returns the last trace, the
    memory (with accumulated reflections), and the number of attempts used.
    """
    memory = Memory()
    plan = make_plan(task, planner)
    trace = Trace(task=task)

    for attempt in range(1, max_attempts + 1):
        trace = react_attempt(task, llm, memory, plan=plan, max_steps=max_steps)
        if trace.succeeded and success_check(trace.answer or ""):
            return trace, memory, attempt
        # Failure -> reflect and retry with the lesson in long-term memory.
        memory.add(reflect(task, trace, reflector))

    return trace, memory, max_attempts


# --------------------------------------------------------------------------
# Optional Ollama backend (graceful when absent), same as W12.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# The ten-problem suite: LONG-TERM memory carried ACROSS problems
# --------------------------------------------------------------------------
def run_suite(problems, llm, *, reflector=None, planner=None,
              max_attempts: int = 2, max_steps: int = 6, carry_memory: bool = True):
    """Work through a list of problems, keeping ONE memory for the whole run.

    This is the difference between "retry until it works" and *learning*. With
    `carry_memory=True` a lesson written after problem 2 is in the prompt for
    problem 3, so the interesting number is not the final score but how many
    problems are solved on the FIRST attempt as the run goes on.

    Returns a list of per-problem dicts.
    """
    from problems import evaluate

    memory = Memory()
    results = []
    for prob in problems:
        if not carry_memory:
            memory = Memory()
        plan = make_plan(prob.question, planner)
        first_ok, attempts, trace, feedback = None, 0, None, ""
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            trace = react_attempt(prob.question, llm, memory, plan, max_steps)
            ok, feedback = evaluate(prob, trace.answer)
            if first_ok is None:
                first_ok = ok
            if ok:
                break
            # The evaluator's message is the external feedback the reflection
            # is written from; without it the lesson is just "that failed".
            memory.add(reflect_with_feedback(prob.question, trace, feedback, reflector))
        results.append({
            "pid": prob.pid, "solved": ok, "first_try": bool(first_ok),
            "attempts": attempts, "answer": trace.answer if trace else None,
            "memory_size": len(memory.notes),
        })
    return results


def reflect_with_feedback(task: str, trace: Trace, feedback: str,
                          reflector: Optional[LLM] = None) -> str:
    """A reflection that has the evaluator's message to work from."""
    actions = ", ".join(f"{s.tool}[{s.tool_input}]" for s in trace.steps if s.tool)
    if reflector is None:
        return (f"On '{task[:48]}...' I answered {trace.answer!r} and was told: "
                f"{feedback} I used: {actions or 'no tools'}. "
                "Next time call calc for every arithmetic step instead of doing it "
                "in my head, and search for any fact I am unsure of.")
    prompt = (
        "You failed a task. Write ONE short lesson (max 25 words) that would help "
        "on similar problems in future. Be concrete about tool use.\n"
        f"Task: {task}\nWhat you did: {actions or 'no tools'}\n"
        f"Evaluator said: {feedback}\nLesson:"
    )
    return reflector(prompt).strip()
