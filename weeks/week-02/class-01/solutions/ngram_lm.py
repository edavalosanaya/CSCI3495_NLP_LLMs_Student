#!/usr/bin/env python3
"""W2C1 reference solution, N-gram Bard.

Run:
    python weeks/week-02/class-01/solutions/ngram_lm.py
"""
from __future__ import annotations
import math
import random
from collections import defaultdict

BOS, EOS = "<s>", "</s>"

CORPUS = [
    "the cat sat on the mat",
    "the cat ran on the hat",
    "i do not like green eggs and ham",
    "i do not like them sam i am",
    "the fish sat on a dish",
    "the cat in the hat sat on the cat",
    "a fish on a dish is a fish i wish",
    "i am sam sam i am",
    "the rat sat on the cat",
    "i like green eggs i like ham",
]


def tokenize(sentence: str) -> list[str]:
    return sentence.lower().split()


def pad(tokens: list[str], n: int) -> list[str]:
    return [BOS] * (n - 1) + tokens + [EOS]


def vocab(sentences: list[str]) -> set[str]:
    v = {EOS}
    for s in sentences:
        v.update(tokenize(s))
    return v


def iter_predictions(n: int, sentences: list[str]):
    """Yield every (context, word) pair the model is asked to predict."""
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(n - 1, len(toks)):
            yield tuple(toks[i - n + 1 : i]), toks[i]


def count_ngrams(sentences: list[str], n: int) -> dict:
    ngram = defaultdict(int)
    context = defaultdict(int)
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i : i + n])
            ngram[gram] += 1
            context[gram[:-1]] += 1
    return {
        "n": n,
        "vocab": vocab(sentences),
        "ngram": dict(ngram),
        "context": dict(context),
    }


def prob(model: dict, context: tuple, word: str) -> float:
    v = len(model["vocab"])
    c_ctx = model["context"].get(context, 0)
    c_gram = model["ngram"].get(context + (word,), 0)
    return (c_gram + 1) / (c_ctx + v)


def generate(model: dict, n: int, max_len: int = 20, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    words = sorted(model["vocab"])  # sorted for determinism
    context = (BOS,) * (n - 1)
    out: list[str] = []
    for _ in range(max_len):
        weights = [prob(model, context, w) for w in words]
        nxt = rng.choices(words, weights=weights, k=1)[0]
        if nxt == EOS:
            break
        out.append(nxt)
        if n > 1:
            context = (context + (nxt,))[1:]
    return out


def perplexity(model: dict, n: int, sentences: list[str]) -> float:
    log_sum = 0.0
    count = 0
    for context, word in iter_predictions(n, sentences):
        log_sum += math.log(prob(model, context, word))
        count += 1
    if count == 0:
        return float("inf")
    return math.exp(-log_sum / count)




def _demo() -> None:
    held_out = ["the cat sat on the hat"]
    train = [s for s in CORPUS]
    print("=" * 60)
    print("N-gram Bard, train, babble, and score with perplexity")
    print("=" * 60)
    for n in (1, 2, 3):
        model = count_ngrams(train, n)
        sample = " ".join(generate(model, n, seed=7))
        pp = perplexity(model, n, held_out)
        label = {1: "unigram", 2: "bigram", 3: "trigram"}[n]
        print(f"\n[{label}]  perplexity on held-out = {pp:7.2f}")
        print(f"   babble: {sample}")
    print("\nLower perplexity = better. Notice fluency rise as n grows.")


if __name__ == "__main__":
    _demo()
