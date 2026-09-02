"""W7C2 reference solution, scaling-behavior scoring core."""
from __future__ import annotations

import string

TASKS = [
    {"q": "What is 2 + 2? Answer with just the number.", "answer": "4"},
    {"q": "What is the capital of France? One word.", "answer": "paris"},
    {"q": "How many days are in a week? Just the number.", "answer": "7"},
    {"q": "What color do you get mixing blue and yellow? One word.", "answer": "green"},
    {"q": "What is 10 minus 3? Just the number.", "answer": "7"},
]


def normalize(text: str) -> str:
    return text.strip().strip(string.punctuation + " ").lower()


def is_correct(model_output: str, target: str) -> bool:
    return normalize(target) in normalize(model_output)


def accuracy(outputs: list[str], targets: list[str]) -> float:
    if len(outputs) == 0:
        # No questions asked is not the same as every question right.
        return 0.0

    correct = 0
    for i in range(len(outputs)):
        if is_correct(outputs[i], targets[i]):
            correct = correct + 1

    return correct / len(outputs)


def scaling_trend(results: dict[str, float]) -> bool:
    # The caller promises these are already ordered smallest model first.
    vals = list(results.values())

    for i in range(len(vals) - 1):
        if vals[i] > vals[i + 1]:
            # A bigger model did worse than the one before it.
            return False

    return True


def _demo() -> None:
    targets = [t["answer"] for t in TASKS]
    small = ["4", "Lyon", "7", "green", "six"]
    large = ["4", "Paris.", "7 days", "Green", "7"]
    acc_small = accuracy(small, targets)
    acc_large = accuracy(large, targets)
    print(f"small-model accuracy: {acc_small:.2f}")
    print(f"large-model accuracy: {acc_large:.2f}")
    trend = scaling_trend({"small": acc_small, "large": acc_large})
    print(f"accuracy non-decreasing with size? {trend}")


if __name__ == "__main__":
    _demo()
