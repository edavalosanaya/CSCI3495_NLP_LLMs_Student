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


def top_k_filter(dist: dict[str, float], k: int) -> dict[str, float]:
    kept = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k])
    z = sum(kept.values())
    return {t: v / z for t, v in kept.items()} if z > 0 else kept


def top_p_filter(dist: dict[str, float], p: float) -> dict[str, float]:
    ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    kept: dict[str, float] = {}
    cum = 0.0
    for tok, prob in ordered:
        kept[tok] = prob
        cum += prob
        if cum >= p:
            break
    z = sum(kept.values())
    return {t: v / z for t, v in kept.items()} if z > 0 else kept


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
