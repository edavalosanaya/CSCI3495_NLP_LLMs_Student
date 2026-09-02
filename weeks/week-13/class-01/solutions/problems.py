"""W13C1: a ten-problem benchmark for the math-solver agent, plus the evaluator."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    pid: str
    question: str
    answer: float
    needs: str          # which tools a correct solution uses


PROBLEMS: list[Problem] = [
    Problem("P1", "A class has 23 students and each needs 4 pencils. "
                  "How many pencils are needed in total?", 92, "calc"),
    Problem("P2", "A rectangular garden is 14 metres by 9 metres. "
                  "What is its area in square metres?", 126, "calc"),
    Problem("P3", "A train covers 240 km in 3 hours. "
                  "What is its average speed in km per hour?", 80, "calc"),
    Problem("P4", "You buy 7 books that cost 12.50 dollars each. "
                  "What is the total cost in dollars?", 87.5, "calc"),
    Problem("P5", "A restaurant bill is 64 dollars and you leave a 15 percent tip. "
                  "How many dollars is the tip?", 9.6, "calc"),
    Problem("P6", "A tank holds 450 litres and is two fifths full. "
                  "How many litres are in it?", 180, "calc"),
    Problem("P7", "What is the population of the capital of France, doubled?",
            4200000, "search + calc"),
    Problem("P8", "What is the square root of 1764?", 42, "calc"),
    Problem("P9", "You invest 1000 dollars at 5 percent compound interest for 3 years. "
                  "What is the final amount in dollars?", 1157.625, "calc"),
    Problem("P10", "A car uses 8 litres of fuel per 100 km. "
                   "How many litres does it use over 350 km?", 28, "calc"),
]

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """Pull the RESULT out of an agent's free-text answer.

    The last number, not the first: models answer "23 * 4 = 92" and the thing
    they are claiming is 92. Reading the first number scores that as 23 and
    marks a correct answer wrong, which then teaches the agent a false lesson.
    """
    nums = _NUM.findall(text.replace(",", "") if text else "")
    return float(nums[-1]) if nums else None


def evaluate(problem: Problem, answer: str | None) -> tuple[bool, str]:
    """The external evaluator. Returns (correct, feedback_for_the_agent).

    The feedback is deliberately specific. Reflexion's whole premise is that a
    richer signal turns into a better lesson, and you can watch that happen by
    swapping the messages here for a bare "wrong" and rerunning the suite.
    """
    if not answer:
        return False, "You produced no final answer. Always end with finish[<number>]."
    got = parse_number(answer)
    if got is None:
        return False, (f"Your answer {answer!r} contains no number. "
                       "Finish with the numeric result only, e.g. finish[92].")
    if abs(got - problem.answer) <= max(1e-6, abs(problem.answer) * 1e-6):
        return True, "Correct."
    return False, (f"Wrong: you answered {got:g} but the correct answer is "
                   f"{problem.answer:g}. Recompute with the calculator instead of "
                   "estimating, and look up any fact you do not know.")


if __name__ == "__main__":   # sanity-check every answer by construction
    import math
    checks = {
        "P1": 23 * 4, "P2": 14 * 9, "P3": 240 / 3, "P4": 7 * 12.5,
        "P5": 64 * 0.15, "P6": 450 * 2 / 5, "P7": 2_100_000 * 2,
        "P8": math.sqrt(1764), "P9": 1000 * 1.05 ** 3, "P10": 8 / 100 * 350,
    }
    for p in PROBLEMS:
        assert abs(checks[p.pid] - p.answer) < 1e-9, (p.pid, checks[p.pid], p.answer)
        print(f"{p.pid:4s} {p.answer:>12g}  ({p.needs})")
    print("all 10 answers verified")
