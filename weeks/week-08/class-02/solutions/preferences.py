"""W8C2 reference solution, be the reward model (Bradley-Terry fit)."""
from __future__ import annotations

import math

PREFERENCES: list[tuple[str, str]] = [
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("B", "C"),
    ("B", "D"),
    ("C", "D"),
]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def neg_log_likelihood(
    scores: dict[str, float], prefs: list[tuple[str, str]]
) -> float:
    total = 0.0
    for w, l in prefs:
        p = sigmoid(scores[w] - scores[l])
        # Clamp to avoid log(0).
        p = min(max(p, 1e-12), 1.0)
        total += -math.log(p)
    return total / len(prefs)


def fit_reward_model(
    prefs: list[tuple[str, str]], lr: float = 0.5, steps: int = 500
) -> dict[str, float]:
    items = sorted({x for pair in prefs for x in pair})
    scores = {x: 0.0 for x in items}
    for _ in range(steps):
        grad = {x: 0.0 for x in items}
        for w, l in prefs:
            # d/ds of -log sigmoid(s_w - s_l):  s_w gets +(1-p), s_l gets -(1-p)
            p = sigmoid(scores[w] - scores[l])
            push = 1.0 - p
            grad[w] += push
            grad[l] -= push
        for x in items:
            scores[x] += lr * grad[x] / len(prefs)
        # Re-center: scores are only identifiable up to an additive constant.
        mean = sum(scores.values()) / len(scores)
        for x in items:
            scores[x] -= mean
    return scores


if __name__ == "__main__":
    scores = fit_reward_model(PREFERENCES)
    ranking = sorted(scores, key=scores.get, reverse=True)
    print("Learned reward-model scores (higher = more preferred):")
    for r in ranking:
        print(f"  {r}: {scores[r]:+.3f}")
    print("Implied ranking:", " > ".join(ranking))
