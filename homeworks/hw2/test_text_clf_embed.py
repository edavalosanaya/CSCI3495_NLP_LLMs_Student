"""Tests for HW2, text classification & word embeddings.

Runs against the student's starter file by default. To check the reference
solution (used by the course test sweep):

    HW2_FROM=solution  python -m pytest homeworks/hw2 -q

The suite gracefully SKIPS while the starter still raises NotImplementedError.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE / "solutions" / "text_clf_embed.py"
    if os.environ.get("HW2_FROM") == "solution"
    else _HERE / "text_clf_embed.py"
)
_spec = importlib.util.spec_from_file_location("hw2_under_test", _SRC)
m = importlib.util.module_from_spec(_spec)
sys.modules["hw2_under_test"] = m
_spec.loader.exec_module(m)


def _implemented() -> bool:
    try:
        m.tokenize("hi")
        m.cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="HW2 not implemented yet (fill in the TODOs)"
)


# --------------------------------------------------------------------------
# A small, separable sentiment dataset (deterministic, offline).
# --------------------------------------------------------------------------
TRAIN_DOCS = [
    "i love this movie it is great",
    "what a great and wonderful film i love it",
    "fantastic acting i really love this",
    "great great great wonderful",
    "i hate this movie it is terrible",
    "what an awful and boring film i hate it",
    "terrible acting i really hate this",
    "awful awful awful boring",
]
TRAIN_LABELS = ["pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"]


# --------------------------------------------------------------------------
# Part A, preprocessing & Naive Bayes
# --------------------------------------------------------------------------
def test_tokenize():
    assert m.tokenize("I LOVE NLP!!!") == ["i", "love", "nlp"]


def test_nb_fit_attributes():
    clf = m.NaiveBayesClassifier().fit(TRAIN_DOCS, TRAIN_LABELS)
    assert clf.classes_ == ["neg", "pos"]
    assert "love" in clf.vocab_ and "hate" in clf.vocab_
    # balanced classes -> equal priors
    assert math.isclose(clf.log_prior_["pos"], math.log(0.5), rel_tol=1e-12)
    # likelihood arrays align with vocab and are proper log-probs (<= 0)
    arr = clf.log_likelihood_["pos"]
    assert arr.shape == (len(clf.vocab_),)
    assert np.all(arr <= 0.0)


def test_nb_likelihood_normalizes():
    # P(w|c) over the vocab must sum to 1 for each class.
    clf = m.NaiveBayesClassifier().fit(TRAIN_DOCS, TRAIN_LABELS)
    for c in clf.classes_:
        total = float(np.exp(clf.log_likelihood_[c]).sum())
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_nb_predicts_training_set():
    clf = m.NaiveBayesClassifier().fit(TRAIN_DOCS, TRAIN_LABELS)
    preds = clf.predict(TRAIN_DOCS)
    assert preds == TRAIN_LABELS


def test_nb_generalizes_to_held_out():
    clf = m.NaiveBayesClassifier().fit(TRAIN_DOCS, TRAIN_LABELS)
    assert clf.predict(["i love this wonderful great film"]) == ["pos"]
    assert clf.predict(["this awful boring terrible movie"]) == ["neg"]


def test_nb_handles_oov_words():
    clf = m.NaiveBayesClassifier().fit(TRAIN_DOCS, TRAIN_LABELS)
    # Out-of-vocabulary words are ignored, not crashing.
    out = clf.predict(["zzzz love wonderful qqqq"])
    assert out == ["pos"]


def test_precision_recall_f1():
    y_true = ["pos", "pos", "neg", "neg", "pos"]
    y_pred = ["pos", "neg", "neg", "pos", "pos"]
    # positive=pos: TP=2 (idx0,4), FP=1 (idx3), FN=1 (idx1)
    p, r, f = m.precision_recall_f1(y_true, y_pred, "pos")
    assert math.isclose(p, 2 / 3, rel_tol=1e-9)
    assert math.isclose(r, 2 / 3, rel_tol=1e-9)
    assert math.isclose(f, 2 / 3, rel_tol=1e-9)


def test_f1_zero_when_no_positive_preds():
    p, r, f = m.precision_recall_f1(["pos", "neg"], ["neg", "neg"], "pos")
    assert p == 0.0 and r == 0.0 and f == 0.0


# --------------------------------------------------------------------------
# Part B, embeddings
# A tiny 2-D embedding space engineered so the analogy holds:
#   gender axis (x) and royalty axis (y).
# --------------------------------------------------------------------------
EMB = {
    "man":   np.array([1.0, 0.0]),
    "woman": np.array([-1.0, 0.0]),
    "king":  np.array([1.0, 1.0]),
    "queen": np.array([-1.0, 1.0]),
    "boy":   np.array([0.9, -0.2]),
    "girl":  np.array([-0.9, -0.2]),
    "apple": np.array([0.05, -3.0]),
}


def test_cosine_basic():
    a = np.array([1.0, 0.0])
    assert math.isclose(m.cosine_similarity(a, a), 1.0, rel_tol=1e-12)
    assert math.isclose(
        m.cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0,
        abs_tol=1e-12,
    )


def test_cosine_zero_vector():
    assert m.cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == 0.0


def test_nearest_neighbors_excludes_self_and_orders():
    nn = m.nearest_neighbors("man", EMB, k=2)
    words = [w for w, _ in nn]
    assert "man" not in words
    assert len(nn) == 2
    # similarities are sorted descending
    sims = [s for _, s in nn]
    assert sims == sorted(sims, reverse=True)
    # 'boy' and 'king' share the +x direction with 'man' -> closest
    assert "boy" in words or "king" in words


def test_analogy_king_man_woman_queen():
    # king - man + woman ~= queen
    result = m.analogy("man", "king", "woman", EMB, k=1)
    assert result[0][0] == "queen"


def test_analogy_excludes_inputs():
    result = m.analogy("man", "king", "woman", EMB, k=5)
    returned = {w for w, _ in result}
    assert returned.isdisjoint({"man", "king", "woman"})


def test_bias_score_sign():
    # 'king' should associate more with the male group than the female group.
    score = m.bias_score("king", ["man", "boy"], ["woman", "girl"], EMB)
    assert score > 0.0
    score2 = m.bias_score("queen", ["man", "boy"], ["woman", "girl"], EMB)
    assert score2 < 0.0
