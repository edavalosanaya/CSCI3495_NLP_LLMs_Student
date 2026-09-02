#!/usr/bin/env python3
"""W2C2 starter, Naive Bayes sentiment classifier. See README.md."""
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
    """GIVEN. Lowercase words, punctuation dropped."""
    return re.findall(r"[a-z']+", text.lower())


def count_corpus(docs: list[str], labels: list[str]) -> tuple[Counter, dict, set]:
    """GIVEN. All the counting: docs per class, word counts per class, vocabulary."""
    class_docs = Counter(labels)
    word_counts = {c: Counter() for c in CLASSES}
    vocab = set()
    for doc, y in zip(docs, labels):
        toks = tokenize(doc)
        word_counts[y].update(toks)
        vocab.update(toks)
    return class_docs, word_counts, vocab


def log_prior(n_docs_in_class: int, n_docs: int, n_classes: int) -> float:
    """The log of the smoothed class prior, log P(c).

    Args:
        n_docs_in_class: how many training documents carry this class label.
        n_docs: how many training documents there are in total.
        n_classes: how many classes exist (2 here, pos and neg).

    Returns:
        A negative float. A prior is a probability below 1, so its log is
        below 0.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   P(c) is in README section 2. Smoothing is why two of the three
    #   arguments appear in the formula at all.
    #
    raise NotImplementedError


def log_likelihood(count_w_c: int, total_words_in_c: int, vocab_size: int) -> float:
    """The log likelihood of one word under one class, log P(w | c), smoothed.

    Args:
        count_w_c: how many times word w appears in class c's training text.
            Zero is normal and has to keep working.
        total_words_in_c: how many word tokens class c's training text holds
            in total, counting repeats.
        vocab_size: how many distinct words the whole training set uses, |V|.

    Returns:
        A negative float.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   P(w | c) is in README section 2. Smoothing touches the top AND the
    #   bottom of that fraction, and vocab_size is why.
    #
    raise NotImplementedError


def score(model: dict, tokens: list[str]) -> dict:
    """Score one tokenized document under every class.

    Args:
        model: the trained model returned by train_nb, holding
            "classes":        the class names, as a tuple
            "vocab":          every word seen anywhere in training, as a set
            "log_prior":      class -> log P(c), from your Step 1
            "log_likelihood": class -> word -> log P(word | c), from Step 2
        tokens: one document, already run through tokenize.

    Returns:
        A class -> score mapping, one entry per class in model["classes"].
        The scores are not probabilities and do not sum to 1; only their
        order matters, which is all predict needs.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   This is the arg max formula in README section 2, minus the arg max.
    #
    #   for each class the model knows:
    #       start a running total at that class's log prior
    #       for each token in the document:
    #           if the model never saw that token, ignore it
    #           otherwise add its log likelihood for this class
    #       record the running total under that class's name
    #   hand back the class-to-total mapping
    #
    #   Add the logs. Multiplying the probabilities is what underflows.
    #
    raise NotImplementedError


def train_nb(docs: list[str], labels: list[str]) -> dict:
    """GIVEN. Counts the corpus, then calls your log_prior / log_likelihood."""
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
    """GIVEN. The argmax: whichever class scored higher."""
    scores = score(model, tokens)
    return max(scores, key=scores.get)


def prf(gold: list[str], pred: list[str], target: str = "pos") -> dict:
    """GIVEN. Precision, recall and F1 for the target class (0.0 if undefined)."""
    tp = sum(1 for g, p in zip(gold, pred) if p == target and g == target)
    fp = sum(1 for g, p in zip(gold, pred) if p == target and g != target)
    fn = sum(1 for g, p in zip(gold, pred) if p != target and g == target)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def _demo() -> None:
    """GIVEN. Trains on TRAIN, predicts TEST, reports precision/recall/F1."""
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
        print("sentiment.py is not finished yet: fill in the next TODO in this file, then re-run.")
