"""W13C2 reference solution: the evaluation harness."""
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
    out: dict[str, list[Result]] = {}
    for name, fn in strategies.items():
        rows = []
        for p in problems:
            result = evaluate_one(p, name, fn, llm)
            rows.append(result)
            if progress:
                if result.correct:
                    mark = "ok"
                else:
                    mark = ". "
                print(f"  {name:10s} {p.pid} {mark}", flush=True)
        out[name] = rows
    return out


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
    by_b = {}
    for r in b:
        by_b[r.pid] = r
    a_only = b_only = same = 0
    for ra in a:
        rb = by_b.get(ra.pid)
        if rb is None:
            continue
        if ra.correct and not rb.correct:
            a_only += 1
        elif rb.correct and not ra.correct:
            b_only += 1
        else:
            same += 1
    return a_only, b_only, same


def rank_key(row: tuple) -> tuple:
    """Sort key for a leaderboard row: success descending, then cost ascending."""
    name, success, calls = row
    return (-success, calls)


def leaderboard(matrix: dict[str, list[Result]]) -> list[tuple[str, float, float]]:
    """(strategy, success_rate, avg_calls), best first; ties broken by cost.

    Cost is in the sort on purpose. If two strategies tie on accuracy the one
    that used fewer model calls is the better engineering answer, and a
    leaderboard that hides cost will always crown the most expensive entry.
    """
    rows = []
    for name, rs in matrix.items():
        rows.append((name, success_rate(rs), avg_calls(rs)))

    # Highest success first; when two tie, the cheaper one wins.
    rows.sort(key=rank_key)
    return rows


def format_leaderboard(matrix: dict[str, list[Result]]) -> str:
    n = len(next(iter(matrix.values()))) if matrix else 0
    lines = [f"{'rank':<5}{'strategy':<12}{'success':>10}{'solved':>9}{'calls/task':>12}"]
    for i, (name, sr, ac) in enumerate(leaderboard(matrix), 1):
        lines.append(f"{i:<5}{name:<12}{sr:>9.0%}{f'{round(sr * n)}/{n}':>9}{ac:>12.1f}")
    return "\n".join(lines)
