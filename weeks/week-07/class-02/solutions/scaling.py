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
    if not outputs:
        return 0.0
    correct = sum(is_correct(o, t) for o, t in zip(outputs, targets))
    return correct / len(outputs)


def scaling_trend(results: dict[str, float]) -> bool:
    vals = list(results.values())
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


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
