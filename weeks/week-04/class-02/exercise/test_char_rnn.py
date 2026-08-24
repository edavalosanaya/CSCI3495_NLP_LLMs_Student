"""Tests for W4C2 char_rnn.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-04/class-02/exercise/test_char_rnn.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  CHARRNN_FROM=solution
"""
import importlib.util
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


def _implemented() -> bool:
    try:
        stoi, _ = cr.build_vocab(["abc"])
        cr.make_training_pairs("abc", stoi)
        model = cr.CharRNN(len(stoi))
        model(torch.tensor([[0, 1]]))
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="char_rnn not implemented yet (fill in the TODOs)"
)


def test_step0_vocab_contains_end_marker():
    stoi, itos = cr.build_vocab(["abc"])
    assert cr.END in stoi
    assert itos[stoi["a"]] == "a"


def test_step2_training_pairs_shift_and_end():
    stoi, itos = cr.build_vocab(["abc"])
    xin, yt = cr.make_training_pairs("abc", stoi)
    assert [itos[i.item()] for i in xin] == ["a", "b", "c"]
    assert [itos[i.item()] for i in yt] == ["b", "c", cr.END]


def test_step1_forward_shapes():
    stoi, _ = cr.build_vocab(cr.NAMES)
    model = cr.CharRNN(len(stoi))
    logits, h = model(torch.tensor([[0, 1, 2]]))
    assert logits.shape == (1, 3, len(stoi))


def test_step3_sample_returns_string_without_end_marker():
    torch.manual_seed(0)
    stoi, itos = cr.build_vocab(cr.NAMES)
    model = cr.CharRNN(len(stoi))
    name = cr.sample(model, stoi, itos, seed="t", max_len=15)
    assert isinstance(name, str)
    assert cr.END not in name
    assert name.startswith("t")


def test_step4_training_reduces_loss():
    torch.manual_seed(cr.SEED)
    stoi, _ = cr.build_vocab(cr.NAMES)
    model = cr.CharRNN(len(stoi))
    history = cr.train(model, cr.NAMES, stoi, epochs=60)
    assert history[-1] < history[0]
