"""Four agent strategies, fully implemented. You EVALUATE these, you do not write them."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tools import calculator

LLM = Callable[[str], str]

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ACTION = re.compile(r"Action:\s*([A-Za-z_]+)\s*\[")


def last_number(text: str) -> Optional[float]:
    """The result a free-text answer is claiming is the LAST number in it."""
    nums = _NUM.findall((text or "").replace(",", ""))
    return float(nums[-1]) if nums else None


def parse_action(text: str) -> Optional[tuple[str, str]]:
    """`Action: tool[input]`, counting brackets so nested ones survive."""
    m = _ACTION.search(text)
    if not m:
        return None
    depth, out, i = 1, [], m.end()
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return m.group(1), "".join(out).strip()
        out.append(ch)
        i += 1
    return None


@dataclass
class Run:
    answer: Optional[float]
    calls: int = 0                       # model calls, the cost of this run
    steps: int = 0                       # tool steps taken
    attempts: int = 1
    trace: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
NAIVE_PROMPT = "{q}\n\nAnswer with just the final number, nothing else."


def naive(question: str, llm: LLM, **_) -> Run:
    text = llm(NAIVE_PROMPT.format(q=question))
    return Run(answer=last_number(text), calls=1, trace=[text.strip()[:200]])


# --------------------------------------------------------------------------
COT_PROMPT = ("{q}\n\nLet's think step by step. Work through it, then write the "
              "final answer on its own last line as a plain number.")


def cot(question: str, llm: LLM, lessons: str = "", **_) -> Run:
    text = llm((lessons + "\n" if lessons else "") + COT_PROMPT.format(q=question))
    return Run(answer=last_number(text), calls=1, trace=[text.strip()[:400]])


# --------------------------------------------------------------------------
REACT_HEADER = """You solve math word problems by reasoning and by calling a
calculator. Each turn, write your reasoning and then ONE action:

Thought: <reason about the next step, as long as you need>
Action: <tool>[<input>]

Tools:
- calc[expression]   arithmetic, e.g. calc[1200 * 1.5]. Use ** for powers.
- finish[number]     give the final number and stop.

Reason freely in Thought, but put every calculation through calc: that is the
one thing the calculator is better at than you are. When you have the number
the question asked for, you MUST end with finish[<number>]. An answer you never
finish does not count.

Example
Task: Josh buys a house for $80,000 and puts $50,000 into repairs, which
increases its value by 150%. How much profit did he make?
Thought: His total outlay is the purchase price plus the repairs.
Action: calc[80000 + 50000]
Observation: 130000
Thought: The value went up by 150% of $80,000, so I add that to the original.
Action: calc[80000 + 80000 * 1.5]
Observation: 200000
Thought: Profit is the new value minus what he put in.
Action: calc[200000 - 130000]
Observation: 70000
Thought: That is the profit.
Action: finish[70000]
"""


def react(question: str, llm: LLM, max_steps: int = 8, lessons: str = "", **_) -> Run:
    prompt = REACT_HEADER + (f"\n{lessons}\n" if lessons else "") + f"\nTask: {question}\n"
    run = Run(answer=None)
    last: Optional[tuple[str, str]] = None
    repeats = 0
    for _ in range(max_steps):
        text = llm(prompt)
        run.calls += 1
        act = parse_action(text)
        if act is None:
            prompt += "Observation: Error: no valid Action. Use Action: tool[input].\n"
            run.trace.append("(malformed)")
            continue
        tool, arg = act
        if tool == "finish":
            run.answer = last_number(arg)
            run.trace.append(f"finish[{arg}]")
            return run
        if (tool, arg) == last:
            # One repeat is a stumble, two in a row is a loop. Breaking on the
            # first cost this strategy real answers: the model often repeats a
            # step once and then moves on.
            repeats += 1
            if repeats >= 2:
                run.trace.append("(looping, stopping)")
                break
            prompt += ("Observation: You already did that and got the same "
                       "result. Move on to the next step, or finish.\n")
            continue
        repeats = 0
        last = (tool, arg)
        obs = calculator(arg) if tool == "calc" else f"Error: unknown tool '{tool}'"
        run.steps += 1
        run.trace.append(f"{tool}[{arg}] -> {obs}")
        prompt += f"{text.strip()}\nObservation: {obs}\n"
    # No explicit finish: fall back to the last good observation, as in W12C2.
    for entry in reversed(run.trace):
        if "->" in entry and "Error" not in entry:
            run.answer = last_number(entry.split("->")[-1])
            break
    return run


# --------------------------------------------------------------------------
def make_reflexion(base: Callable, name: str = "Reflexion"):
    """Wrap ANY base strategy in the Reflexion loop.

    Reflexion is not a competitor to CoT or ReAct, it is a layer you put on top
    of one of them: attempt, get external feedback, write the lesson down, try
    again with the lesson in the prompt. Wrapping the strongest base is the
    whole trick, and wrapping a weak one just buys you a slightly less weak
    agent, which the leaderboard will show you plainly.
    """

    def reflexion(question: str, llm: LLM, feedback_fn=None, max_attempts: int = 2,
                  **kw) -> Run:
        lessons, total = "", Run(answer=None)
        for attempt in range(1, max_attempts + 1):
            run = base(question, llm, lessons=lessons, **kw)
            total.calls += run.calls
            total.steps += run.steps
            total.trace += [f"[try {attempt}] " + t for t in run.trace]
            total.answer, total.attempts = run.answer, attempt
            if feedback_fn is None:
                return total          # no evaluator: nothing to reflect on
            ok, message = feedback_fn(run.answer)
            if ok:
                return total
            lessons = (
                "You have already tried this problem and got it WRONG.\n"
                f"- Your previous answer: {run.answer}\n"
                f"- Feedback: {message}\n"
                "Do not repeat that answer. Re-read the question, check what it "
                "is actually asking for, and redo each calculation.\n")
        return total

    reflexion.__name__ = name
    return reflexion


# Each Reflexion entry sits directly under the baseline it wraps, so the
# leaderboard reads as two before/after pairs rather than four rivals.
STRATEGIES = {
    "Naive": naive,
    "CoT": cot,
    "Reflexion+CoT": make_reflexion(cot, "Reflexion+CoT"),
    "ReAct": react,
    "Reflexion+ReAct": make_reflexion(react, "Reflexion+ReAct"),
}

# Which baseline each Reflexion variant is layered on, for the lift table.
REFLEXION_PAIRS = [("Reflexion+CoT", "CoT"), ("Reflexion+ReAct", "ReAct")]

COURSE_MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:1.5b")


def make_ollama_llm(model: str = COURSE_MODEL, stop_at_observation: bool = True) -> LLM:
    import ollama

    client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))

    def llm(prompt: str) -> str:
        opts = {"temperature": 0.0, "num_predict": 400}
        if stop_at_observation:
            opts["stop"] = ["Observation:"]
        return client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                           options=opts)["message"]["content"]

    return llm


def ollama_available(model: str = COURSE_MODEL) -> bool:
    try:
        import ollama

        client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))
        return any(model.split(":")[0] in m.get("model", "")
                   for m in client.list().get("models", []))
    except Exception:  # noqa: BLE001
        return False
