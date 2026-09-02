"""W13C2 lab: the evaluation harness (starter).

The four strategies are given (`strategies.py`). This file is the part that
decides what "better" means, which is the actual skill:

  * is_correct     numeric match with a tolerance, not string equality
  * evaluate_one   run one strategy on one problem, record answer AND cost
  * run_matrix     every strategy x every problem
  * success_rate / avg_calls
  * paired_wins    per-problem head to head, the honest small-sample comparison
  * leaderboard    rank by success, break ties by cost
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

DATA = Path(__file__).resolve().parent.parent / "exercise" / "data" / "gsm8k_mini.jsonl"


@dataclass(frozen=True)
class Problem:
    pid: str
    question: str
    answer: float


def load_problems(path: Path = DATA, limit: Optional[int] = None) -> list[Problem]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    probs = [Problem(r["id"], r["question"], float(r["answer"])) for r in rows]
    return probs[:limit] if limit else probs


@dataclass
class Result:
    pid: str
    strategy: str
    predicted: Optional[float]
    correct: bool
    calls: int = 0
    steps: int = 0
    attempts: int = 1
    trace: list[str] = field(default_factory=list)


def is_correct(predicted: Optional[float], gold: float, tol: float = 1e-4) -> bool:
    """Numeric match with a tolerance.

    Not string equality: the strategies legitimately return 18, 18.0 and
    "18.00", and a string comparison would score three identical answers three
    different ways. Tolerance is relative for big numbers so 1596 vs 1596.0001
    does not fail.
    """
    if predicted is None:
        return False
    return abs(predicted - gold) <= max(tol, abs(gold) * tol)


def evaluate_one(problem: Problem, name: str, fn: Callable, llm, **kw) -> Result:
    """Run one strategy on one problem. Reflexion gets the evaluator to react to."""
    if name.startswith("Reflexion"):
        def feedback(ans):
            if is_correct(ans, problem.answer):
                return True, "Correct."
            return False, (f"Wrong: you answered {ans}. Recheck each arithmetic "
                           "step with the calculator.")
        kw = {**kw, "feedback_fn": feedback}
    run = fn(problem.question, llm, **kw)
    return Result(problem.pid, name, run.answer,
                  is_correct(run.answer, problem.answer),
                  run.calls, run.steps, run.attempts, run.trace)


def run_matrix(problems: list[Problem], strategies: dict, llm,
               progress: bool = False) -> dict[str, list[Result]]:
    """Every strategy against every problem. Returns {strategy: [Result, ...]}."""
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   Every strategy against every problem.
    #
    #   1. for each (name, fn) in strategies.items():
    #         for each problem: evaluate_one(p, name, fn, llm), collect the Result
    #         if progress: print one line per problem so a slow run shows life
    #   2. return {strategy_name: [Result, ...]}
    #
    raise NotImplementedError


def success_rate(results: list[Result]) -> float:
    return sum(r.correct for r in results) / len(results) if results else 0.0


def avg_calls(results: list[Result]) -> float:
    return sum(r.calls for r in results) / len(results) if results else 0.0


def paired_wins(a: list[Result], b: list[Result]) -> tuple[int, int, int]:
    """Per-problem head to head: (a_only, b_only, both_or_neither).

    On 20 problems a 3-point gap in success rate is 0.6 of a problem, which is
    noise. What is actually informative is how often A solved something B did
    not, so this pairs them by problem id instead of comparing two averages.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   Compare two strategies PROBLEM BY PROBLEM, not average against average.
    #
    #   1. index b by problem id: {r.pid: r for r in b}
    #   2. for each result ra in a, find rb with the same pid (skip if absent):
    #         ra correct and rb not -> a_only += 1
    #         rb correct and ra not -> b_only += 1
    #         otherwise             -> same += 1
    #   3. return (a_only, b_only, same)
    #
    raise NotImplementedError


def leaderboard(matrix: dict[str, list[Result]]) -> list[tuple[str, float, float]]:
    """(strategy, success_rate, avg_calls), best first; ties broken by cost.

    Cost is in the sort on purpose. If two strategies tie on accuracy the one
    that used fewer model calls is the better engineering answer, and a
    leaderboard that hides cost will always crown the most expensive entry.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   1. for each (name, results), build (name, success_rate(rs), avg_calls(rs))
    #   2. sort by success rate DESCENDING, then by average calls ASCENDING
    #         key=lambda r: (-r[1], r[2])
    #   3. return the sorted list
    #
    #   Cost is in the sort on purpose: on a tie, the cheaper strategy wins.
    #
    raise NotImplementedError


def format_leaderboard(matrix: dict[str, list[Result]]) -> str:
    n = len(next(iter(matrix.values()))) if matrix else 0
    lines = [f"{'rank':<5}{'strategy':<12}{'success':>10}{'solved':>9}{'calls/task':>12}"]
    for i, (name, sr, ac) in enumerate(leaderboard(matrix), 1):
        lines.append(f"{i:<5}{name:<12}{sr:>9.0%}{f'{round(sr * n)}/{n}':>9}{ac:>12.1f}")
    return "\n".join(lines)
