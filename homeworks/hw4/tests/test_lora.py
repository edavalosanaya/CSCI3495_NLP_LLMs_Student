"""Tests for HW4 (LoRA / PEFT).

By default these run against the student's ``homeworks/hw4/lora.py``. To test
the reference solution (used by the course sweep), set:
    HW4_FROM=solution
The module is skipped gracefully if the student has not implemented it yet.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "lora.py"
    if os.environ.get("HW4_FROM") == "solution"
    else _HERE.parent / "lora.py"
)
_spec = importlib.util.spec_from_file_location("hw4_lora_under_test", _SRC)
lora = importlib.util.module_from_spec(_spec)
sys.modules["hw4_lora_under_test"] = lora
_spec.loader.exec_module(lora)


def _implemented() -> bool:
    try:
        lora.LoRALinear(nn.Linear(4, 4), r=2, alpha=4.0)
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not _implemented(), reason="HW4 lora.py not implemented yet (fill in the TODOs)"
)


# --- Task 1: LoRALinear ----------------------------------------------------
def test_step1_lora_starts_as_identity():
    """B = 0 at init, so the LoRA layer must equal the base layer on step 0."""
    torch.manual_seed(0)
    base = nn.Linear(8, 5)
    layer = lora.LoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(layer(x), base(x), atol=1e-6)


def test_step1_lora_freezes_base_and_trains_adapters():
    base = nn.Linear(8, 5)
    layer = lora.LoRALinear(base, r=2, alpha=4.0)
    assert layer.base.weight.requires_grad is False
    assert layer.base.bias.requires_grad is False
    assert layer.A.requires_grad is True
    assert layer.B.requires_grad is True
    assert layer.A.shape == (2, 8)
    assert layer.B.shape == (5, 2)


def test_step1_lora_scaling_and_update():
    """After perturbing A and B, output shifts by scaling * B @ A @ x."""
    torch.manual_seed(1)
    base = nn.Linear(6, 4)
    layer = lora.LoRALinear(base, r=2, alpha=8.0)
    with torch.no_grad():
        layer.A.copy_(torch.randn_like(layer.A))
        layer.B.copy_(torch.randn_like(layer.B))
    x = torch.randn(7, 6)
    expected = base(x) + (8.0 / 2) * (x @ layer.A.t()) @ layer.B.t()
    assert torch.allclose(layer(x), expected, atol=1e-6)


# --- Task 2: injection + counting -----------------------------------------
def test_step2_inject_replaces_named_linear():
    model = lora.TinyClassifier(in_dim=16, hidden=32, n_classes=2)
    lora.inject_lora(model, target="fc1", r=4, alpha=8.0)
    assert isinstance(model.fc1, lora.LoRALinear)
    assert isinstance(model.fc2, nn.Linear) and not isinstance(model.fc2, lora.LoRALinear)


def test_step3_trainable_param_fraction_is_small():
    model = lora.TinyClassifier(in_dim=16, hidden=32, n_classes=2)
    lora.inject_lora(model, target="fc1", r=4, alpha=8.0)
    lora.inject_lora(model, target="fc2", r=4, alpha=8.0)
    # Head is unfrozen by default; freeze it so only LoRA adapters train.
    for p in model.head.parameters():
        p.requires_grad_(False)
    trainable, total = lora.count_trainable_parameters(model)
    assert 0 < trainable < total
    assert trainable / total < 0.5  # LoRA = a small fraction of all params


# --- Task 3: toy data + training ------------------------------------------
def test_step4_toy_dataset_deterministic():
    X1, y1 = lora.make_toy_dataset(n=64, in_dim=16, seed=0)
    X2, y2 = lora.make_toy_dataset(n=64, in_dim=16, seed=0)
    assert X1.shape == (64, 16) and y1.shape == (64,)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)
    assert set(y1.tolist()) <= {0, 1}


def test_step5_lora_training_reduces_loss():
    """The headline check: training ONLY LoRA adapters lowers the loss."""
    torch.manual_seed(0)
    model = lora.TinyClassifier(in_dim=16, hidden=32, n_classes=2)
    lora.inject_lora(model, target="fc1", r=4, alpha=8.0)
    lora.inject_lora(model, target="fc2", r=4, alpha=8.0)
    X, y = lora.make_toy_dataset(n=256, in_dim=16, seed=0)
    history = lora.train_lora(model, X, y, epochs=40, lr=0.05, seed=0)
    assert len(history) == 40
    # Loss should decrease meaningfully from start to end.
    assert history[-1] < history[0] - 0.01


def test_step5_only_adapters_changed_after_training():
    """Frozen base weights must be unchanged after training."""
    torch.manual_seed(0)
    model = lora.TinyClassifier(in_dim=16, hidden=32, n_classes=2)
    before = model.fc1.weight.detach().clone()
    lora.inject_lora(model, target="fc1", r=4, alpha=8.0)
    X, y = lora.make_toy_dataset(n=128, in_dim=16, seed=0)
    lora.train_lora(model, X, y, epochs=10, lr=0.05, seed=0)
    after = model.fc1.base.weight.detach()
    assert torch.allclose(before, after, atol=1e-7)


# --- Cross-check against Hugging Face PEFT (offline, no download) ----------
def test_step1_peft_lora_matches_concept():
    """Sanity-check our mental model against the real `peft` library.

    We build a tiny module with an nn.Linear, wrap it with peft's LoraConfig,
    and confirm PEFT also (a) freezes the base weight and (b) starts as an
    identity (lora_B initialized to zero, like our implementation).
    """
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(0)
    net = Net()
    x = torch.randn(2, 8)
    base_out = net(x).detach().clone()

    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none")
    peft_model = get_peft_model(net, cfg)

    # PEFT also starts as identity (lora_B = 0).
    assert torch.allclose(peft_model(x), base_out, atol=1e-5)

    # Base weight is frozen; only LoRA params are trainable.
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    assert 0 < n_trainable < n_total


def test_step3_count_trainable_parameters_exact_numbers():
    """The counts are checkable by hand, so check them by hand."""
    base = nn.Linear(4, 3, bias=True)                   # 12 weights + 3 biases = 15
    trainable, total = lora.count_trainable_parameters(base)
    assert (trainable, total) == (15, 15)

    wrapped = lora.LoRALinear(base, r=2, alpha=4.0)     # A 2x4=8, B 3x2=6 -> 14 trainable
    trainable, total = lora.count_trainable_parameters(wrapped)
    assert trainable == 14, "only A and B train; the base must be frozen"
    assert total == 29, "the frozen base still counts toward the total"


def test_step2_inject_lora_reaches_nested_modules():
    """The replacement is recursive, so a target inside a submodule is still found."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
    lora.inject_lora(model, target="0", r=2)
    assert isinstance(model[0], lora.LoRALinear)
    assert isinstance(model[1][0], lora.LoRALinear), "nested target was not replaced"
