"""Tests for W4C1 mlp_classifier.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  MLP_FROM=solution
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
    _HERE.parent / "solutions" / "mlp_classifier.py"
    if os.environ.get("MLP_FROM") == "solution"
    else _HERE / "mlp_classifier.py"
)
_spec = importlib.util.spec_from_file_location("mlp_under_test", _SRC)
mc = importlib.util.module_from_spec(_spec)
sys.modules["mlp_under_test"] = mc
_spec.loader.exec_module(mc)


def test_step0_vocab_has_unk_at_zero():
    vocab = mc.build_vocab(["hello world"])
    assert vocab["<unk>"] == 0
    assert set(["hello", "world"]).issubset(vocab)


def test_step1_embed_document_shape_and_average():
    vocab = mc.build_vocab(["a b"])
    emb = mc.nn.Embedding(len(vocab), 4)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(len(vocab) * 4, dtype=torch.float).reshape(len(vocab), 4))
    vec = mc.embed_document("a b", vocab, emb)
    assert vec.shape == (4,)
    expected = (emb.weight[vocab["a"]] + emb.weight[vocab["b"]]) / 2
    assert torch.allclose(vec, expected)


def test_step2_forward_returns_two_logits():
    model = mc.MLP(in_dim=16, hidden=8)
    out = model(torch.zeros(3, 16))
    assert out.shape == (3, 2)


def test_step3_training_learns():
    torch.manual_seed(0)
    vocab = mc.build_vocab([t for t, _ in mc.TRAIN])
    emb = mc.nn.Embedding(len(vocab), 16)
    model = mc.MLP(in_dim=16, hidden=8)
    history = mc.train(model, emb, vocab, mc.TRAIN)
    assert history[-1] < history[0]
    assert mc.accuracy(model, emb, vocab, mc.TRAIN) == 1.0
