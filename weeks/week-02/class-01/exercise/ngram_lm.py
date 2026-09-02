#!/usr/bin/env python3
"""W2C1 starter, build an N-gram Bard. See README.md."""
from __future__ import annotations
import random
from collections import defaultdict
import math

BOS, EOS = "<s>", "</s>"

# A tiny, playful corpus (nursery-rhyme / Dr.-Seuss flavored). Small on purpose.
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
    """Lowercase whitespace tokenizer (the corpus is already clean)."""
    return sentence.lower().split()


def pad(tokens: list[str], n: int) -> list[str]:
    """Add (n-1) BOS tokens at the front and one EOS at the end."""
    return [BOS] * (n - 1) + tokens + [EOS]


def vocab(sentences: list[str]) -> set[str]:
    """The set of word types, including EOS (BOS is never predicted)."""
    v = {EOS}
    for s in sentences:
        v.update(tokenize(s))
    return v


def count_ngrams(sentences: list[str], n: int) -> dict:
    """GIVEN. One counting pass over the padded corpus.

    Returns {"n", "vocab", "ngram": {(w1..wn): count}, "context": {(w1..w_{n-1}): count}}.
    """
    ngram: dict[tuple, int] = defaultdict(int)
    context: dict[tuple, int] = defaultdict(int)
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(n - 1, len(toks)):
            gram = tuple(toks[i - n + 1 : i + 1])
            ngram[gram] += 1
            context[gram[:-1]] += 1
    return {"n": n, "vocab": vocab(sentences), "ngram": dict(ngram), "context": dict(context)}


def iter_predictions(n: int, sentences: list[str]):
    """GIVEN. Yields every (context, word) pair the model is asked to predict."""
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(n - 1, len(toks)):
            yield tuple(toks[i - n + 1 : i]), toks[i]


def prob(model: dict, context: tuple, word: str) -> float:
    """Add-one smoothed P(word | context) under an n-gram model.

    Args:
        model: the dict from count_ngrams, holding
            "vocab":   every word that can be predicted, including </s>
            "ngram":   full n-gram tuple -> how often it was seen
            "context": the same tuple minus its last word -> how often seen
            Both counts are plain dicts: a tuple never seen is absent, not 0.
        context: the n-1 words before the one being predicted, as a tuple.
        word: the single word whose probability you want.

    Returns:
        A float in (0, 1). Never exactly 0, even for a pair never seen
        together, and never 1: that is what the smoothing buys.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   P(w | context) is in README section 2. One line of arithmetic, no loop.
    #
    #   Both counts come out of the model with .get(..., 0), because an unseen
    #   tuple is missing from the dict rather than stored as zero. The full
    #   n-gram's key is the context tuple with the word appended.
    #
    raise NotImplementedError


def generate(model: dict, n: int, max_len: int = 20, seed: int = 0) -> list[str]:
    """GIVEN. Walks the chain from <s> to </s>, sampling each word from prob()."""
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
    """Perplexity of held-out sentences: how surprised the model is, per word.

    Args:
        model: the dict from count_ngrams.
        n: the model's order, so the padding matches how it was trained.
        sentences: held-out text, NOT the training corpus.

    Returns:
        A float >= 1. Lower is better. Return float("inf") when there is
        nothing to predict, since an average over zero tokens is undefined.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The formula is in README section 2.
    #
    #   iter_predictions(n, sentences) hands you each (context, word) pair,
    #   already padded, so there is no windowing to do here.
    #
    #   add up the log of prob(...) for every pair, counting the pairs
    #   if you counted none, the answer is infinity
    #   otherwise return e to the power of minus the average log
    #
    #   Work in logs and exponentiate once at the end. Multiplying the
    #   probabilities of a long sentence underflows to zero.
    #
    raise NotImplementedError


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
    print("\nLower perplexity = better.")


if __name__ == "__main__":
    try:
        _demo()
    except NotImplementedError:
        print("ngram_lm.py is not finished yet: fill in the next TODO in this file, then re-run.")
