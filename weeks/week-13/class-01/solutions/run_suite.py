"""W13C1: does long-term memory actually make the agent better?

Runs the ten-problem suite twice with the SAME model and the SAME tools, and
changes exactly one thing: whether reflections written on problem N are still
in the prompt at problem N+1.

    docker compose -f docker/docker-compose.yml run --rm --no-deps \\
      -e OLLAMA_HOST=http://host.docker.internal:11434 \\
      course python weeks/week-13/class-01/solutions/run_suite.py

Needs Ollama with COURSE_MODEL pulled (default qwen2.5:1.5b); prints a message
and exits cleanly if it is missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import run_suite  # noqa: E402
from problems import PROBLEMS  # noqa: E402

COURSE_MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:1.5b")


def make_llm(model: str):
    import ollama

    client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))

    def llm(prompt: str) -> str:
        return client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "stop": ["Observation:"]},
        )["message"]["content"]

    return llm


def available(model: str) -> bool:
    try:
        import ollama

        client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))
        names = [m.get("model", "") for m in client.list().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:  # noqa: BLE001
        return False


def report(label: str, results: list[dict]) -> None:
    first = sum(r["first_try"] for r in results)
    final = sum(r["solved"] for r in results)
    print(f"\n{label}")
    print(f"  solved on the FIRST attempt : {first}/10")
    print(f"  solved within 2 attempts    : {final}/10")
    marks = " ".join(f"{r['pid']}:{'OK' if r['first_try'] else ' .'}" for r in results)
    print(f"  first-attempt timeline      : {marks}")


def main() -> int:
    if not available(COURSE_MODEL):
        print(f"Ollama model '{COURSE_MODEL}' not found. Start Ollama and run:")
        print(f"  ollama pull {COURSE_MODEL}")
        return 0
    llm = make_llm(COURSE_MODEL)
    print(f"model: {COURSE_MODEL}   problems: {len(PROBLEMS)}")

    report("A. memory RESET between problems (reflection helps within a problem only)",
           run_suite(PROBLEMS, llm, carry_memory=False, max_attempts=2))
    report("B. memory CARRIED across problems (long-term memory)",
           run_suite(PROBLEMS, llm, carry_memory=True, max_attempts=2))

    print("\nThe only difference between A and B is whether a lesson written on an")
    print("earlier problem is still in the prompt for a later one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
