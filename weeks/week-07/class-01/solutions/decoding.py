"""W7C1 reference solution, decoding strategies from scratch."""
from __future__ import annotations

import math
import random


def apply_temperature(logits: dict[str, float], temperature: float) -> dict[str, float]:
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
    ordered = sorted(dist.items(), key=by_probability)

    kept = {}
    for token, probability in ordered[:k]:
        kept[token] = probability

    return renormalize(kept)


def top_p_filter(dist: dict[str, float], p: float) -> dict[str, float]:
    ordered = sorted(dist.items(), key=by_probability)

    kept = {}
    running_total = 0.0
    for token, probability in ordered:
        # Keep this token FIRST, then check. Stopping before the token that
        # crosses p would leave the kept mass below p.
        kept[token] = probability
        running_total = running_total + probability
        if running_total >= p:
            break

    return renormalize(kept)


def sample(dist: dict[str, float], seed: int | None = None) -> str:
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
    _demo()
