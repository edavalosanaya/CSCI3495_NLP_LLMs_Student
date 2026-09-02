"""W7C1 starter, decoding strategies, from scratch.

You implement the core math of decoding on a toy next-token distribution (pure
Python, no model needed). Then `playground.py` lets you feel the SAME knobs on a
real local LLM via Ollama (skips cleanly if Ollama isn't running).

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-07/class-01/exercise/test_decoding.py -k step1 -q

A "distribution" here is a dict {token: probability} that sums to ~1.0.
"""
from __future__ import annotations

import math
import random


def apply_temperature(logits: dict[str, float], temperature: float) -> dict[str, float]:
    """Apply temperature to LOGITS and return a normalized probability dict.

    p_i = softmax(logit_i / T). Lower T -> sharper; higher T -> flatter.
    For T very close to 0, treat it as greedy: all mass on the argmax token.
    """
    if temperature < 1e-6:
        # Greedy: all mass on the argmax.
        best = max(logits, key=logits.get)
        return {t: (1.0 if t == best else 0.0) for t in logits}
    scaled = {t: v / temperature for t, v in logits.items()}
    m = max(scaled.values())  # for numerical stability
    exps = {t: math.exp(v - m) for t, v in scaled.items()}
    z = sum(exps.values())
    return {t: e / z for t, e in exps.items()}


def greedy(dist: dict[str, float]) -> str:
    """Return the single highest-probability token (ties: any is fine)."""
    return max(dist, key=dist.get)


def by_probability(item: tuple) -> float:
    """Sort key for (token, probability) pairs: most likely first."""
    token, probability = item
    return -probability

def renormalize(kept: dict[str, float]) -> dict[str, float]:
    """Scale the kept probabilities so they add up to 1 again.

    Dropping the tail of a distribution leaves the rest summing to less than 1,
    which is no longer a distribution. Dividing each by the new total fixes it.
    """
    total = 0.0
    for probability in kept.values():
        total = total + probability

    if total == 0.0:
        return kept

    scaled = {}
    for token in kept:
        scaled[token] = kept[token] / total
    return scaled


def top_k_filter(dist: dict[str, float], k: int) -> dict[str, float]:
    """Keep the k most likely tokens and throw the rest of the tail away.

    Args:
        dist: token -> probability, summing to 1.
        k: how many tokens to keep. k larger than the vocabulary is not an
            error; it just keeps everything.

    Returns:
        A new dict of at most k entries, summing to 1 again. The input is not
        modified.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   order the tokens with the given by_probability, and take the first k
    #   hand the survivors to the given renormalize
    #
    #   The dividing is the step people skip. After you delete the tail, what
    #   is left no longer sums to 1, and it is no longer a distribution.
    #
    raise NotImplementedError


def top_p_filter(dist: dict[str, float], p: float) -> dict[str, float]:
    """Keep the smallest group of top tokens that together reach probability p.

    Args:
        dist: token -> probability, summing to 1.
        p: the mass to cover, between 0 and 1. Unlike top-k, how many tokens
            this keeps depends on how confident the distribution is.
        
    Returns:
        A new dict summing to 1. Never empty: even a p of 0 keeps the single
        most likely token, because returning nothing to sample from is worse
        than returning one thing.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   order the tokens by probability, most likely first
    #   walk that order, keeping each token and adding up the mass so far,
    #       and stop once the running total has reached p
    #   hand the survivors to renormalize, as in step 1
    #
    #   Keep the token that TAKES you past p, then stop. Stopping before it
    #   leaves you under p, which is the off-by-one to watch for.
    #
    raise NotImplementedError


def sample(dist: dict[str, float], seed: int | None = None) -> str:
    """Sample a token from `dist` according to its probabilities.

    Use random.Random(seed) for determinism in tests.
    """
    rng = random.Random(seed)
    r = rng.random()
    cum = 0.0
    last = None
    for tok, prob in dist.items():
        last = tok
        cum += prob
        if r <= cum:
            return tok
    return last  # floating-point fallback


def _demo() -> None:
    logits = {"sunny": 2.0, "cloudy": 1.2, "cold": 0.6, "nice": 0.1, "banana": -3.0}
    cold_dist = apply_temperature(logits, 0.5)
    hot_dist = apply_temperature(logits, 1.5)
    print("temp=0.5 ->", {k: round(v, 3) for k, v in cold_dist.items()})
    print("temp=1.5 ->", {k: round(v, 3) for k, v in hot_dist.items()})
    base = apply_temperature(logits, 1.0)
    print("greedy   ->", greedy(base))
    print("top-2    ->", {k: round(v, 3) for k, v in top_k_filter(base, 2).items()})
    print("top-p .8 ->", {k: round(v, 3) for k, v in top_p_filter(base, 0.8).items()})


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        _demo()
    except NotImplementedError:
        print("decoding.py is not finished yet: fill in the next TODO in this file, then re-run.")
