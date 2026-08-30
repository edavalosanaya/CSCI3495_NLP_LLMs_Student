#!/usr/bin/env python3
"""W2C2 starter, Naive Bayes sentiment classifier from scratch.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-02/class-02/exercise/test_sentiment.py -k step1 -q

When all four steps are done, the demo runs:
    python weeks/week-02/class-02/exercise/sentiment.py

CPU-only, deterministic, no network. Build the model yourself (no sklearn).
"""
from __future__ import annotations
import math
import re

# Tiny hand-labeled movie-review snippets. label "pos" / "neg".
TRAIN = [
    ("a wonderful and moving film i loved it", "pos"),
    ("brilliant acting and a great story", "pos"),
    ("the best movie i have seen this year", "pos"),
    ("funny charming and beautifully shot", "pos"),
    ("a delightful and clever masterpiece", "pos"),
    ("boring and predictable a total waste", "neg"),
    ("terrible acting and a dull story", "neg"),
    ("the worst movie i have seen this year", "neg"),
    ("a slow and forgettable mess", "neg"),
    ("dreadful plot and awful pacing", "neg"),
]

TEST = [
    ("a great and moving story i loved it", "pos"),
    ("brilliant and beautifully shot", "pos"),
    ("a dull and boring waste of time", "neg"),
    ("the worst and most forgettable mess", "neg"),
    ("clever and funny but a bit slow", "pos"),
]

CLASSES = ("pos", "neg")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer."""
    return re.findall(r"[a-z']+", text.lower())


def train_nb(docs: list[str], labels: list[str]) -> dict:
    """Train a multinomial Naive Bayes model with add-one smoothing.

    Return a dict like:
        {
          "classes": ("pos", "neg"),
          "log_prior": {c: log P(c)},
          "log_likelihood": {c: {word: log P(word | c)}},
          "vocab": set_of_words,
          "class_total": {c: total_word_tokens_in_c},
        }
    Use add-one smoothing so unseen words get nonzero probability:
        P(w | c) = (count(w,c) + 1) / (class_total[c] + |V|)
    For words not stored explicitly, score() will compute the smoothed
    fallback 1 / (class_total[c] + |V|).

    Smooth the priors too so a class with zero training docs is still valid:
        P(c) = (class_docs[c] + 1) / (N + num_classes)
    With two classes these priors still sum to 1.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


def score(model: dict, tokens: list[str]) -> dict:
    """Return {class: log-probability} for the tokenized doc.

    log P(c) + sum_w log P(w | c). Words outside the vocabulary are skipped.
    Words in the vocab but unseen in class c use the smoothed fallback.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    # Do NOT skip an in-vocab word just because class c never saw it.
    raise NotImplementedError


def predict(model: dict, tokens: list[str]) -> str:
    """Return the class with the highest score."""
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    scores = score(model, tokens)
    return max(scores, key=scores.get)


def prf(gold: list[str], pred: list[str], target: str = "pos") -> dict:
    """Precision, recall, F1 for the target class.

    Return {"precision": p, "recall": r, "f1": f1}. If a denominator is 0,
    define that metric as 0.0.
    """
    # GIVEN (STEP 4): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    tp = sum(1 for g, p in zip(gold, pred) if p == target and g == target)
    fp = sum(1 for g, p in zip(gold, pred) if p == target and g != target)
    fn = sum(1 for g, p in zip(gold, pred) if p != target and g == target)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def _demo() -> None:
    docs = [d for d, _ in TRAIN]
    labels = [y for _, y in TRAIN]
    model = train_nb(docs, labels)

    gold, pred = [], []
    print("=" * 60)
    print("Sentiment Showdown, Naive Bayes from scratch")
    print("=" * 60)
    for text, y in TEST:
        p = predict(model, tokenize(text))
        gold.append(y)
        pred.append(p)
        mark = "OK " if p == y else "XX "
        print(f"  [{mark}] gold={y}  pred={p}   {text!r}")

    m = prf(gold, pred, target="pos")
    print(f"\n  pos-class  precision={m['precision']:.2f}  "
          f"recall={m['recall']:.2f}  F1={m['f1']:.2f}")


if __name__ == "__main__":
    try:
        _demo()
    except NotImplementedError:
        print("sentiment.py is not implemented yet, fill in the TODOs, then re-run.")
