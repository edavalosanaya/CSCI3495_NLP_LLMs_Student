"""HW2 starter, text classification & word embeddings.

Two parts:
  Part A, a Multinomial Naive Bayes sentiment classifier with add-1 smoothing,
           implemented from scratch in NumPy, plus precision/recall/F1.
  Part B, word embeddings: cosine similarity, nearest neighbors, the classic
           analogy task (king - man + woman ~= queen), and a tiny bias probe.

Run the tests:
    docker compose -f docker/docker-compose.yml run --rm course \
        python -m pytest homeworks/hw2 -q

Allowed libraries: numpy (and the Python standard library). Do NOT use
scikit-learn's classifiers for Part A, implement the math yourself. (You MAY
use sklearn elsewhere for your own experiments, but the graded functions below
must be your own NumPy.)
"""
# Each TODO below names its README step. Check one step with:
#     python -m pytest homeworks/hw2 -q -k step3      (or step1, step2, ...)
# and the whole assignment with:
#     python -m pytest homeworks/hw2 -q

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric/apostrophe word tokens."""
    # TODO (STEP 1): return _TOKEN_RE.findall(text.lower())
    raise NotImplementedError


# ===========================================================================
# Part A, Multinomial Naive Bayes (from scratch)
# ===========================================================================
class NaiveBayesClassifier:
    """Multinomial Naive Bayes with add-1 (Laplace) smoothing.

    After ``fit``:
      - self.classes_     : sorted list of class labels
      - self.vocab_       : sorted list of vocabulary tokens (the feature set)
      - self.log_prior_   : dict[label -> log P(class)]
      - self.log_likelihood_ : dict[label -> np.ndarray of shape (|V|,)]
                               giving log P(word | class) aligned with vocab_
    """

    def __init__(self):
        self.classes_: list = []
        self.vocab_: list[str] = []
        self._index: dict[str, int] = {}
        self.log_prior_: dict = {}
        self.log_likelihood_: dict = {}

    def fit(self, docs: list[str], labels: list) -> "NaiveBayesClassifier":
        """Train on parallel lists of raw documents and their labels.

        Steps (use natural log everywhere):
          1. classes_ = sorted set of labels.
          2. vocab_   = sorted set of all tokens across all docs; build
             self._index mapping token -> column index.
          3. log_prior_[c]   = log( count(docs in c) / total docs ).
          4. For each class c, accumulate a word-count vector over its docs.
             With add-1 smoothing and |V| = len(vocab_):
                 P(w | c) = (count(w, c) + 1) / (total_words_in_c + |V|)
             Store log P(w | c) as a NumPy array aligned with vocab_.
        """
        # TODO (STEP 2): implement
        raise NotImplementedError

    def _features(self, doc: str) -> np.ndarray:
        """Return a length-|V| count vector for one document (OOV words ignored)."""
        # TODO (STEP 3): build a counts vector using self._index
        raise NotImplementedError

    def predict_log_scores(self, doc: str) -> dict:
        """Return {class: log_prior + sum(count_w * log P(w|class))} for a doc."""
        # TODO (STEP 3): implement
        raise NotImplementedError

    def predict(self, docs: list[str]) -> list:
        """Return the argmax class for each document."""
        # TODO (STEP 3): implement
        raise NotImplementedError


def precision_recall_f1(y_true: list, y_pred: list, positive) -> tuple[float, float, float]:
    """Binary precision, recall, and F1 for the given ``positive`` label.

    precision = TP / (TP + FP)   (0.0 if denominator is 0)
    recall    = TP / (TP + FN)   (0.0 if denominator is 0)
    f1        = 2 * P * R / (P + R)  (0.0 if P + R == 0)
    """
    # TODO (STEP 4): implement
    raise NotImplementedError


# ===========================================================================
# Part B, Word embeddings
# ===========================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 if either is zero."""
    # TODO (STEP 5): implement (dot / (||a|| * ||b||))
    raise NotImplementedError


def nearest_neighbors(
    word: str, embeddings: dict[str, np.ndarray], k: int = 5
) -> list[tuple[str, float]]:
    """Return the ``k`` most cosine-similar words to ``word`` (excluding itself).

    ``embeddings`` maps word -> vector. Result is a list of (word, similarity)
    sorted by similarity descending. Ties broken alphabetically by word.
    """
    # TODO (STEP 6): implement
    raise NotImplementedError


def analogy(
    a: str, b: str, c: str, embeddings: dict[str, np.ndarray], k: int = 1
) -> list[tuple[str, float]]:
    """Solve the analogy 'a is to b as c is to ?'.

    Compute the target vector  v = emb[b] - emb[a] + emb[c]  and return the
    ``k`` nearest words by cosine similarity, EXCLUDING the input words a, b, c.
    Result is a list of (word, similarity) sorted by similarity descending,
    ties broken alphabetically.
    """
    # TODO (STEP 7): implement
    raise NotImplementedError


def bias_score(
    word: str,
    group_a: list[str],
    group_b: list[str],
    embeddings: dict[str, np.ndarray],
) -> float:
    """A simple association-bias probe (mini-WEAT).

    Return the difference between the mean cosine similarity of ``word`` to the
    words in ``group_a`` and its mean similarity to ``group_b``:

        bias = mean_{x in A} cos(word, x) - mean_{y in B} cos(word, y)

    A positive score means ``word`` is more associated with group A. This is the
    core idea behind embedding-bias measurements (e.g. WEAT).
    """
    # TODO (STEP 8): implement
    raise NotImplementedError
