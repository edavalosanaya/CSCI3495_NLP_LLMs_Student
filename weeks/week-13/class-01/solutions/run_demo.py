#!/usr/bin/env python3
"""W13C1 demo, memory + planning + Reflexion, deterministically (no Ollama).

Shows an agent that FAILS its first attempt, writes a reflection to memory,
and SUCCEEDS on the retry, the core Reflexion (Shinn et al., 2023) idea.

    python run_demo.py
"""
from __future__ import annotations

from agent import run_reflexion_agent
from scripted_llm import ReflexiveLLM, const


def main() -> int:
    task = "What is the population of the capital of France, doubled?"

    # Attempt 1 (no lesson yet): the agent guesses without searching -> wrong.
    before = [
        "Thought: I think it's a big number.\nAction: finish[10 million]",
    ]
    # Attempt 2 (after reflection appears in memory): search, then compute.
    after = [
        "Thought: The lesson says search first.\nAction: search[paris]",
        "Thought: ~2.1 million people; double it.\nAction: calc[2100000 * 2]",
        "Thought: Done.\nAction: finish[4200000]",
    ]
    llm = ReflexiveLLM(before, after)

    def success_check(ans: str) -> bool:
        return "4200000" in ans.replace(",", "")

    trace, memory, attempts = run_reflexion_agent(
        task,
        llm,
        success_check=success_check,
        planner=const("1. Find Paris's population.\n2. Double it.\n3. Finish."),
        reflector=None, # use the deterministic fallback reflection
        max_attempts=3,
    )

    print(f"Task: {task}")
    print(f"Attempts used: {attempts}")
    print("\nLong-term memory (reflections):")
    for n in memory.notes:
        print(f"  - {n}")
    print("\nFinal trace:")
    for i, s in enumerate(trace.steps, 1):
        if s.tool == "finish":
            print(f"  [{i}] finish[{s.tool_input}]")
        elif s.tool:
            print(f"  [{i}] {s.tool}[{s.tool_input}] -> {s.observation}")
        else:
            print(f"  [{i}] (no action) {s.observation}")
    ok = trace.succeeded and success_check(trace.answer or "")
    print(f"\nAnswer: {trace.answer!r}  ->  {'CORRECT' if ok else 'WRONG'} after {attempts} attempt(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
