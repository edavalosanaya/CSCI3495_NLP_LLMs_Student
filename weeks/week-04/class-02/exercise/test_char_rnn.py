"""Tests for W4C2 char_rnn.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-04/class-02/exercise/test_char_rnn.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  CHARRNN_FROM=solution
"""
import importlib.util
import inspect
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "char_rnn.py"
    if os.environ.get("CHARRNN_FROM") == "solution"
    else _HERE / "char_rnn.py"
)
_spec = importlib.util.spec_from_file_location("charrnn_under_test", _SRC)
cr = importlib.util.module_from_spec(_spec)
sys.modules["charrnn_under_test"] = cr
_spec.loader.exec_module(cr)


def test_given_vocab_contains_end_marker():
    stoi, itos = cr.build_vocab(["abc"])
    assert cr.END in stoi
    assert itos[stoi["a"]] == "a"


def test_given_training_pairs_shift_and_end():
    stoi, itos = cr.build_vocab(["abc"])
    xin, yt = cr.make_training_pairs("abc", stoi)
    assert [itos[i.item()] for i in xin] == ["a", "b", "c"]
    assert [itos[i.item()] for i in yt] == ["b", "c", cr.END]


def test_step1_rnn_step_matches_the_formula():
    # Everything is 1s, so the two products are just sums: 2 + 3 + 1 = 6,
    # and tanh(6) is 0.99998...
    h_prev = torch.ones(2)
    x_t = torch.ones(3)
    w_h = torch.ones(2, 2)
    w_x = torch.ones(2, 3)
    b = torch.ones(2)
    h = cr.rnn_step(h_prev, x_t, w_h, w_x, b)
    assert h.shape == (2,)
    assert h.tolist() == pytest.approx([0.99998771, 0.99998771], abs=1e-6)


def test_step1_rnn_step_agrees_with_nn_rnn():
    torch.manual_seed(0)
    stoi, _ = cr.build_vocab(cr.NAMES)
    model = cr.CharRNN(len(stoi))
    h_torch, h_mine = cr.compare_one_step(model, stoi, char="t")
    assert float((h_torch - h_mine).abs().max()) < 1e-5


def test_step2_sample_next_returns_a_valid_index():
    torch.manual_seed(0)
    stoi, _ = cr.build_vocab(cr.NAMES)
    logits = torch.zeros(1, 3, len(stoi))
    nxt = cr.sample_next(logits)
    assert isinstance(nxt, int)
    assert 0 <= nxt < len(stoi)


def test_step2_sample_next_draws_the_likely_character_most_often():
    # One character scores far above the rest, so a draw lands there most of
    # the time, but not every time. argmax would return it 40 out of 40.
    torch.manual_seed(0)
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 2] = 3.0
    draws = [cr.sample_next(logits) for _ in range(40)]
    assert draws.count(2) > 20
    assert len(set(draws)) > 1


def test_given_training_reduces_loss():
    torch.manual_seed(cr.SEED)
    stoi, _ = cr.build_vocab(cr.NAMES)
    model = cr.CharRNN(len(stoi))
    history = cr.train(model, cr.NAMES, stoi, epochs=60)
    assert history[-1] < history[0]
