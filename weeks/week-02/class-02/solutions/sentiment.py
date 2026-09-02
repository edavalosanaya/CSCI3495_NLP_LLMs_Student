#!/usr/bin/env python3
"""W2C2 reference solution, Naive Bayes sentiment classifier from scratch."""
from __future__ import annotations
import math
import re
from collections import Counter

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
    return re.findall(r"[a-z']+", text.lower())


def count_corpus(docs: list[str], labels: list[str]) -> tuple[Counter, dict, set]:
    class_docs = Counter(labels)
    word_counts = {c: Counter() for c in CLASSES}
    vocab = set()
    for doc, y in zip(docs, labels):
        toks = tokenize(doc)
        word_counts[y].update(toks)
        vocab.update(toks)
    return class_docs, word_counts, vocab


def log_prior(n_docs_in_class: int, n_docs: int, n_classes: int) -> float:
    return math.log((n_docs_in_class + 1) / (n_docs + n_classes))


def log_likelihood(count_w_c: int, total_words_in_c: int, vocab_size: int) -> float:
    return math.log((count_w_c + 1) / (total_words_in_c + vocab_size))


def score(model: dict, tokens: list[str]) -> dict:
    out = {}
    for c in model["classes"]:
        s = model["log_prior"][c]
        for w in tokens:
            if w in model["vocab"]:
                s += model["log_likelihood"][c][w]
        out[c] = s
    return out


def train_nb(docs: list[str], labels: list[str]) -> dict:
    class_docs, word_counts, vocab = count_corpus(docs, labels)
    v = len(vocab)
    return {
        "classes": CLASSES,
        "vocab": vocab,
        "log_prior": {
            c: log_prior(class_docs[c], len(docs), len(CLASSES)) for c in CLASSES
        },
        "log_likelihood": {
            c: {
                w: log_likelihood(word_counts[c][w], sum(word_counts[c].values()), v)
                for w in vocab
            }
            for c in CLASSES
        },
    }


def predict(model: dict, tokens: list[str]) -> str:
    scores = score(model, tokens)
    return max(scores, key=scores.get)


def prf(gold: list[str], pred: list[str], target: str = "pos") -> dict:
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
    _demo()
