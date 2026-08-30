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
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
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
    # GIVEN (STEP 2): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return max(dist, key=dist.get)


def top_k_filter(dist: dict[str, float], k: int) -> dict[str, float]:
    """Keep only the k highest-probability tokens; renormalize so they sum to 1."""
    # TODO (STEP 3): implement. Check with: pytest -k step3
    raise NotImplementedError


def top_p_filter(dist: dict[str, float], p: float) -> dict[str, float]:
    """Nucleus filter: keep the smallest set of top tokens whose probabilities
    sum to >= p; renormalize. (Always keep at least the top-1 token.)"""
    # TODO (STEP 4): implement. Check with: pytest -k step4
    raise NotImplementedError


def sample(dist: dict[str, float], seed: int | None = None) -> str:
    """Sample a token from `dist` according to its probabilities.

    Use random.Random(seed) for determinism in tests.
    """
    # GIVEN (STEP 5): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
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
