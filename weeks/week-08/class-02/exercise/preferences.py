"""W8C2 starter, be the reward model."""
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
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def neg_log_likelihood(
    scores: dict[str, float], prefs: list[tuple[str, str]]
) -> float:
    """How surprised the model is by the humans' choices, on average.

    Args:
        scores: response -> current scalar score. Only differences between
            scores matter, never their absolute size.
        prefs: (winner, loser) pairs. The order inside each pair IS the label:
            a human said the first one was better.

    Returns:
        The mean over all pairs. Low means winners reliably score above losers.
        Always finite: a probability of exactly 0 would make the log blow up,
        so clamp it away from 0 before taking the log.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The loss is in README section 2.
    #
    #   for each winner/loser pair:
    #       the model's probability the winner wins is the sigmoid of the
    #           winner's score minus the loser's
    #       nudge that probability away from 0 before taking its log
    #       add the negative log of it to a running total
    #   divide by how many pairs there were
    #
    raise NotImplementedError


def fit_reward_model(
    prefs: list[tuple[str, str]], lr: float = 0.5, steps: int = 500
) -> dict[str, float]:
    """Learn a score per response from nothing but pairwise human choices.

    Args:
        prefs: (winner, loser) pairs, the whole training signal.
        lr: gradient-descent step size.
        steps: how many full passes over prefs to run.

    Returns:
        response -> score, for every response mentioned anywhere in prefs.
        The scores are re-centred to mean 0, because only DIFFERENCES are
        identifiable: adding 5 to every score describes the same preferences.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   Plain gradient descent, written out by hand. No autograd.
    #
    #   collect every distinct response mentioned in prefs, and start all their
    #       scores at zero
    #   repeat `steps` times:
    #       start a fresh gradient of zero for every response
    #       for each winner/loser pair, work out how wrong the model currently
    #           is: one minus its probability the winner wins. Push the winner
    #           up by that amount and the loser down by the same amount
    #       move every score along its gradient, scaled by lr and averaged over
    #           the number of pairs
    #       subtract the mean score from every score
    #
    #   That last line is not cosmetic. Without it the scores drift together
    #   forever and never settle, because nothing pins down where zero is.
    #
    raise NotImplementedError


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        scores = fit_reward_model(PREFERENCES)
        ranking = sorted(scores, key=scores.get, reverse=True)
        print("Learned reward-model scores (higher = more preferred):")
        for r in ranking:
            print(f"  {r}: {scores[r]:+.3f}")
        print("Implied ranking:", " > ".join(ranking))
    except NotImplementedError:
        print("preferences.py is not finished yet: fill in the next TODO in this file, then re-run.")
