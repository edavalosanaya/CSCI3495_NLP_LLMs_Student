"""W8C2 starter, be the reward model.

Part A (on paper / in the dataset): you label which of two responses is better.
Part B (here): turn those *comparisons* into numeric scores by fitting a tiny
Bradley-Terry reward model, exactly the idea behind RLHF's Stage 2.

A "preference" is a pair where response `w` (winner) was preferred over `l`
(loser). The Bradley-Terry model gives each response a scalar score `s`, and
says the probability that w beats l is sigmoid(s_w - s_l). We fit the scores by
gradient descent to maximize the likelihood of YOUR labels.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-08/class-02/exercise/test_preferences.py -k step1 -q
"""
from __future__ import annotations

import math

# A toy comparison set over 4 candidate responses to one prompt, named A-D.
# Each tuple is (winner, loser): the winner was preferred.
# These encode a clear ranking  A > B > C > D  (try changing them!).
PREFERENCES: list[tuple[str, str]] = [
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("B", "C"),
    ("B", "D"),
    ("C", "D"),
]


def sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


def neg_log_likelihood(
    scores: dict[str, float], prefs: list[tuple[str, str]]
) -> float:
    """Average negative log-likelihood of the preferences under Bradley-Terry.

    For each (w, l): probability w beats l is sigmoid(scores[w] - scores[l]).
    Return the MEAN of -log(that probability) over all preferences.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


def fit_reward_model(
    prefs: list[tuple[str, str]], lr: float = 0.5, steps: int = 500
) -> dict[str, float]:
    """Fit per-response scalar scores by gradient descent on the BT loss.

    Start every score at 0.0. For each (w, l), the gradient of -log sigmoid(s_w - s_l)
    pushes s_w up and s_l down by  (1 - sigmoid(s_w - s_l)).
    After each full pass, re-center scores to mean 0 (scores are only
    meaningful up to a constant). Return the final {response: score} dict.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    raise NotImplementedError


if __name__ == "__main__":
    scores = fit_reward_model(PREFERENCES)
    ranking = sorted(scores, key=scores.get, reverse=True)
    print("Learned reward-model scores (higher = more preferred):")
    for r in ranking:
        print(f"  {r}: {scores[r]:+.3f}")
    print("Implied ranking:", " > ".join(ranking))
