"""Tests for W9C1 lora_lab.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-09/class-01/exercise/test_lora_lab.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  LORA_FROM=solution  (used by the course test sweep).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "lora_lab.py"
    if os.environ.get("LORA_FROM") == "solution"
    else _HERE / "lora_lab.py"
)
_spec = importlib.util.spec_from_file_location("lora_under_test", _SRC)
lab = importlib.util.module_from_spec(_spec)
sys.modules["lora_under_test"] = lab
_spec.loader.exec_module(lab)


def test_step1_base_weight_frozen():
    m = lab.LoRALinear(8, 4, r=4, alpha=8)
    assert m.linear.weight.requires_grad is False


def test_step2_lora_params_trainable():
    m = lab.LoRALinear(8, 4, r=4, alpha=8)
    trainable = {name for name, p in m.named_parameters() if p.requires_grad}
    assert "A" in trainable and "B" in trainable
    # The base weight must NOT be trainable.
    assert "linear.weight" not in trainable


def test_step3_scaling_is_alpha_over_r():
    m = lab.LoRALinear(8, 4, r=4, alpha=8)
    assert m.scaling == pytest.approx(2.0)
    m2 = lab.LoRALinear(8, 4, r=2, alpha=8)
    assert m2.scaling == pytest.approx(4.0)


def test_step4_forward_shape():
    torch.manual_seed(0)
    m = lab.LoRALinear(8, 4, r=4, alpha=8)
    x = torch.randn(5, 8)
    assert m(x).shape == (5, 4)


def test_step4_starts_at_base_model():
    # With B initialized to zero, the LoRA update is zero at init.
    torch.manual_seed(0)
    m = lab.LoRALinear(8, 4, r=4, alpha=8)
    x = torch.randn(3, 8)
    with torch.no_grad():
        base = m.linear(x)
        full = m(x)
    assert torch.allclose(base, full, atol=1e-6)


def test_step4_training_reduces_loss():
    torch.manual_seed(0)
    in_f, out_f, n = 8, 4, 64
    X = torch.randn(n, in_f)
    Y = X @ torch.randn(out_f, in_f).T
    m = lab.LoRALinear(in_f, out_f, r=4, alpha=8)
    losses = lab.train_lora(m, X, Y, steps=200)
    # The adapter should make at least meaningful progress on the toy task.
    assert losses[-1] < losses[0] * 0.9


def test_step4_only_adapter_changed():
    # Base weight values must be identical before and after training.
    torch.manual_seed(0)
    m = lab.LoRALinear(8, 4, r=4, alpha=8)
    before = m.linear.weight.detach().clone()
    X = torch.randn(32, 8)
    Y = X @ torch.randn(4, 8).T
    lab.train_lora(m, X, Y, steps=50)
    assert torch.allclose(before, m.linear.weight.detach())


def test_step5_quantize_error_decreases_with_bits():
    torch.manual_seed(0)
    w = torch.randn(1000)
    e8 = lab.quant_error(w, 8)
    e4 = lab.quant_error(w, 4)
    e2 = lab.quant_error(w, 2)
    # More bits -> less error.
    assert e8 < e4 < e2


def test_step5_quantize_shape_preserved():
    w = torch.randn(7, 3)
    q = lab.quantize(w, 4)
    assert q.shape == w.shape
