#!/usr/bin/env python3
"""W12C1, "Break the Agent" adversarial game (no Ollama required).

You play the role of the LLM. Type the agent's next move each turn:
    Thought: <whatever>
    Action: calc[<expr>]   |   search[<query>]   |   finish[<answer>]

Your mission: try to BREAK the agent loop, make it crash, loop forever,
run unsafe code, or give a wrong Observation. The robust agent should
survive everything you throw at it (budget stops, safe calc, caught errors).
Score = how many distinct failure modes you *fail* to trigger (higher = the
agent is more robust). Then read the agent code and explain WHY each attack
was defended.

This imports the reference agent from the Week-12 Class-2 solution so you can
experiment before you build your own.

Run:
    python weeks/week-12/class-01/exercise/break_the_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the reference agent + tools from W12C2 solutions.
_SOL = Path(__file__).resolve().parents[2] / "week-12" / "class-02" / "solutions"
sys.path.insert(0, str(_SOL))

from agent import run_agent  # noqa: E402

ATTACKS = {
    "infinite loop": "did the agent stop on its own (budget/stuck) rather than hang?",
    "unsafe code": "did calc[] refuse to run __import__/os.system as code?",
    "tool crash": "did a bad tool input (e.g. calc[1/0]) become an Observation, not a crash?",
    "malformed action": "did a missing/garbled Action get a corrective Observation?",
}


class HumanLLM:
    """You are the model: each call reads one Thought+Action from stdin."""

    def __call__(self, transcript: str) -> str:
        print("\n--- transcript so far (tail) ---")
        print("\n".join(transcript.strip().splitlines()[-6:]))
        print("--- your move (end with a blank line) ---")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "":
                break
            lines.append(line)
        return "\n".join(lines) or "Thought: (silence)\nAction: finish[give up]"


def main() -> int:
    print(__doc__)
    print("Attacks to try:")
    for name, q in ATTACKS.items():
        print(f"  - {name}: {q}")
    task = input("\nEnter a task for the agent (or press Enter for a default): ").strip()
    task = task or "Try to break me. What is 12 * 47?"
    trace = run_agent(task, HumanLLM(), max_steps=6)
    print("\n=== RESULT ===")
    print(f"answer={trace.answer!r}  stopped={trace.stopped_reason}  steps={len(trace.steps)}")
    print("Now open ../../week-12/class-02/solutions/agent.py and tools.py and explain")
    print("which guard defended each attack you tried.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
