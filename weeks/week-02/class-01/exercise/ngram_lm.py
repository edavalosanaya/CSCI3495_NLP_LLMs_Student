#!/usr/bin/env python3
"""W2C1 starter, build an N-gram Bard.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-02/class-01/exercise/test_ngram_lm.py -k step1 -q

When every step is done, the demo runs:
    python weeks/week-02/class-01/exercise/ngram_lm.py

Everything is CPU-only and deterministic (we seed the RNG). No network needed.
"""
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
    """Count n-grams and contexts.

    Return a dict:
        {
          "n": n,
          "vocab": set_of_words,          # words that can be PREDICTED (incl. </s>)
          "ngram": {(w1..wn): count},     # full n-gram counts
          "context": {(w1..w_{n-1}): count}  # context counts
        }
    Pad each sentence with pad(tokens, n) before counting.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


def prob(model: dict, context: tuple, word: str) -> float:
    """Add-one (Laplace) smoothed P(word | context).

        (count(context, word) + 1) / (count(context) + V)

    where V = |vocab|. `context` is a tuple of length n-1.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


def generate(model: dict, n: int, max_len: int = 20, seed: int = 0) -> list[str]:
    """Sample a sentence by walking the chain from <s> ... to </s>.

    Start with context = (BOS,) * (n-1). At each step, sample the next word
    from the smoothed distribution over the vocabulary given the context.
    Stop at EOS or after max_len words. Return the generated words
    (WITHOUT the BOS/EOS padding).
    """
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
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
    """Perplexity of the held-out sentences under the model.

        PP = exp( - (1/N) * sum log P(w_t | context) )

    Sum log-probs over every predicted token (the EOS counts; BOS padding
    does not). N is the total number of predicted tokens. Use math.log / math.exp.
    """
    # GIVEN (STEP 4): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    log_sum = 0.0
    count = 0
    for s in sentences:
        toks = pad(tokenize(s), n)
        for i in range(n - 1, len(toks)):
            context = tuple(toks[i - n + 1 : i])
            word = toks[i]
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
    try:
        _demo()
    except NotImplementedError:
        print("ngram_lm.py is not finished yet: fill in the next TODO in this file, then re-run.")
