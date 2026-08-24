"""Tests for HW1, n-gram language models.

Runs against the student's starter file by default. To check the reference
solution (used by the course test sweep), set:

    NGRAM_FROM=solution  python -m pytest homeworks/hw1 -q

The whole suite gracefully SKIPS while the starter still raises
NotImplementedError, so the sweep stays green before students implement.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE / "solutions" / "ngram_lm.py"
    if os.environ.get("NGRAM_FROM") == "solution"
    else _HERE / "ngram_lm.py"
)
_spec = importlib.util.spec_from_file_location("ngram_lm_under_test", _SRC)
lm = importlib.util.module_from_spec(_spec)
sys.modules["ngram_lm_under_test"] = lm
_spec.loader.exec_module(lm)


def _implemented() -> bool:
    try:
        lm.tokenize("hi there")
        lm.NGramLM(2).fit([["a", "b"]])
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="ngram_lm not implemented yet (fill in the TODOs)"
)

# A tiny deterministic corpus reused across tests.
CORPUS = [
    ["i", "am", "sam"],
    ["sam", "i", "am"],
    ["i", "do", "not", "like", "green", "eggs", "and", "ham"],
]


# -- Task 1: preprocessing -------------------------------------------------
def test_tokenize():
    assert lm.tokenize("Hi, NLP!") == ["hi", ",", "nlp", "!"]


def test_tokenize_lowercases_and_counts():
    assert lm.tokenize("NLP is FUN") == ["nlp", "is", "fun"]


def test_sentences_split_and_strip_punct():
    out = lm.sentences("Hi there. NLP rocks!")
    assert out == [["hi", "there"], ["nlp", "rocks"]]


def test_sentences_drop_empty():
    assert lm.sentences("...!!!") == []


# -- Task 2: padding & ngrams ---------------------------------------------
def test_pad_bigram():
    m = lm.NGramLM(n=2)
    assert m.pad(["a", "b"]) == [lm.BOS, "a", "b", lm.EOS]


def test_pad_trigram():
    m = lm.NGramLM(n=3)
    assert m.pad(["a"]) == [lm.BOS, lm.BOS, "a", lm.EOS]


def test_pad_unigram():
    m = lm.NGramLM(n=1)
    assert m.pad(["a", "b"]) == ["a", "b", lm.EOS]


def test_ngrams_bigram():
    m = lm.NGramLM(n=2)
    assert m.ngrams([lm.BOS, "a", lm.EOS]) == [(lm.BOS, "a"), ("a", lm.EOS)]


# -- Task 2: counts & vocab -----------------------------------------------
def test_fit_vocab_includes_specials_excludes_bos():
    m = lm.NGramLM(n=2).fit(CORPUS)
    assert lm.EOS in m.vocab and lm.UNK in m.vocab
    assert lm.BOS not in m.vocab
    assert "sam" in m.vocab


def test_fit_counts_bigram():
    m = lm.NGramLM(n=2).fit(CORPUS)
    # "i am" appears: (i,am) in sent0, (i,am) in sent1 -> 2 times
    assert m.ngram_counts[("i", "am")] == 2
    # context "i" precedes: am, am, do -> count 3
    assert m.context_counts[("i",)] == 3


# -- Task 2: probabilities (add-k) ----------------------------------------
def test_prob_is_distribution_over_vocab():
    m = lm.NGramLM(n=2, k=1.0).fit(CORPUS)
    ctx = ("i",)
    total = sum(m.prob(w, ctx) for w in m.vocab)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_prob_add_k_known_value():
    m = lm.NGramLM(n=2, k=1.0).fit(CORPUS)
    v = len(m.vocab)
    # P(am | i) = (count(i,am)+1) / (count(i)+1*V) = (2+1)/(3+V)
    expected = (2 + 1) / (3 + v)
    assert math.isclose(m.prob("am", ("i",)), expected, rel_tol=1e-12)


def test_prob_unknown_token_maps_to_unk():
    m = lm.NGramLM(n=2, k=1.0).fit(CORPUS)
    # An unseen word must still get nonzero probability via smoothing/UNK.
    assert m.prob("zzz", ("i",)) > 0.0
    # Unknown context word also handled (no KeyError, positive prob).
    assert m.prob("am", ("qqq",)) > 0.0


def test_unigram_prob_distribution():
    m = lm.NGramLM(n=1, k=1.0).fit(CORPUS)
    total = sum(m.prob(w, ()) for w in m.vocab)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


# -- Task 2: sentence logprob & perplexity --------------------------------
def test_sentence_logprob_negative():
    m = lm.NGramLM(n=2).fit(CORPUS)
    lp = m.sentence_logprob(["i", "am", "sam"])
    assert lp < 0.0


def test_sentence_logprob_matches_sum_of_logs():
    m = lm.NGramLM(n=2).fit(CORPUS)
    sent = ["i", "am"]
    padded = m.pad(sent)
    manual = sum(
        math.log(m.prob(g[-1], g[:-1])) for g in m.ngrams(padded)
    )
    assert math.isclose(m.sentence_logprob(sent), manual, rel_tol=1e-12)


def test_perplexity_positive_finite():
    m = lm.NGramLM(n=2).fit(CORPUS)
    pp = m.perplexity(CORPUS)
    assert pp > 1.0 and math.isfinite(pp)


def test_lower_k_lowers_train_perplexity():
    # Less smoothing fits the training data more tightly -> lower perplexity.
    hi = lm.NGramLM(n=2, k=1.0).fit(CORPUS).perplexity(CORPUS)
    lo = lm.NGramLM(n=2, k=0.01).fit(CORPUS).perplexity(CORPUS)
    assert lo < hi


# -- Task 2: generation ----------------------------------------------------
def test_generate_is_reproducible():
    m = lm.NGramLM(n=2).fit(CORPUS)
    a = m.generate(max_len=10, seed=0)
    b = m.generate(max_len=10, seed=0)
    assert a == b


def test_generate_excludes_specials_and_respects_max_len():
    m = lm.NGramLM(n=2).fit(CORPUS)
    out = m.generate(max_len=8, seed=1)
    assert lm.BOS not in out and lm.EOS not in out
    assert len(out) <= 8
