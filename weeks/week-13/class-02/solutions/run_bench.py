"""W13C2: run all four strategies over the GSM8K slice and print the leaderboard.

    docker compose -f docker/docker-compose.yml run --rm --no-deps \\
      -e OLLAMA_HOST=http://host.docker.internal:11434 \\
      course python weeks/week-13/class-02/solutions/run_bench.py [--n 8]

Needs Ollama with COURSE_MODEL pulled (default qwen2.5:1.5b). The full 20
problems x 4 strategies takes a while on a CPU, so pass `--n` for a quick pass
during the lab and run the full suite once at the end.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "exercise"))   # strategies.py, tools.py, data/
sys.path.insert(0, str(_HERE))

from eval_suite import (format_leaderboard, load_problems,  # noqa: E402
                        paired_wins, run_matrix, success_rate)
from strategies import (COURSE_MODEL, REFLEXION_PAIRS, STRATEGIES,  # noqa: E402
                        make_ollama_llm, ollama_available)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="use only the first N problems")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()

    if not ollama_available():
        print(f"Ollama model '{COURSE_MODEL}' not found. Start Ollama and run:")
        print(f"  ollama pull {COURSE_MODEL}")
        return 0

    problems = load_problems(limit=a.n)
    print(f"model: {COURSE_MODEL}   problems: {len(problems)}   "
          f"strategies: {', '.join(STRATEGIES)}")
    t0 = time.time()
    matrix = run_matrix(problems, STRATEGIES, make_ollama_llm(), progress=a.progress)
    print("\n" + format_leaderboard(matrix))
    print(f"\nwall clock: {time.time() - t0:.0f}s")

    print("\nReflexion lift, each variant against the baseline it wraps:")
    for wrapper, base in REFLEXION_PAIRS:
        if wrapper not in matrix or base not in matrix:
            continue
        gained, lost, same = paired_wins(matrix[wrapper], matrix[base])
        sr_w, sr_b = success_rate(matrix[wrapper]), success_rate(matrix[base])
        print(f"  {wrapper:16s} {sr_w:>4.0%}  vs  {base:8s} {sr_b:>4.0%}   "
              f"recovered: {gained:2d}   broke: {lost:2d}   unchanged: {same:2d}")

    print("\nBaselines, paired by problem:")
    for x, y in (("CoT", "Naive"), ("ReAct", "CoT")):
        only_x, only_y, same = paired_wins(matrix[x], matrix[y])
        print(f"  {x:9s} vs {y:9s}  {x} only: {only_x:2d}   {y} only: {only_y:2d}   "
              f"same: {same:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
