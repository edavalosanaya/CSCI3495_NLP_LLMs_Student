"""W12C2 demo: the same questions, asked with and without tools.

    python weeks/week-12/class-02/solutions/run_demo.py

Needs a local Ollama with COURSE_MODEL pulled (default qwen2.5:0.5b); prints a
clear message and exits if it is missing. Everything else in this lab runs
without a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import (COURSE_MODEL, ask_directly, make_ollama_llm,  # noqa: E402
                   ollama_available, run_agent)
from tools import TOOLS  # noqa: E402

MATH_Q = "What is log(3^2 * 16 - 10)?"
DATE_Q = "What is today's date?"
HOT_Q = "How much hotter is it today than yesterday in San Antonio?"


def show(trace) -> None:
    for s in trace.steps:
        if s.tool == "finish":
            print(f"    finish[{s.tool_input}]")
        elif s.tool:
            print(f"    {s.tool}[{s.tool_input}] -> {s.observation}")
        else:
            print(f"    (malformed) -> {s.observation}")
    print(f"    stopped: {trace.stopped_reason}")


def main() -> int:
    if not ollama_available():
        print(f"Ollama model '{COURSE_MODEL}' not found. Start Ollama and run:")
        print(f"  ollama pull {COURSE_MODEL}")
        return 0

    print(f"model: {COURSE_MODEL}\n")
    print("=" * 62)
    print("WITHOUT TOOLS (the model answers from memory)")
    print("=" * 62)
    for q in (MATH_Q, DATE_Q):
        print(f"  {q}\n    {ask_directly(q + ' Answer with just the value.')}\n")

    llm = make_ollama_llm()
    print("=" * 62)
    print("WITH TOOLS (the model may call calc / today / weather / search)")
    print("=" * 62)
    for q in (MATH_Q, DATE_Q):
        print(f"  {q}")
        show(run_agent(q, llm, TOOLS))
        print()

    print("=" * 62)
    print("A QUESTION THAT NEEDS THREE TOOL CALLS")
    print("=" * 62)
    print(f"  {HOT_Q}")
    show(run_agent(HOT_Q, llm, TOOLS, max_steps=8, multi_step=True,
                   require_grounded=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
