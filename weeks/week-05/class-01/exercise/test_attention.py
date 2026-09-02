"""Tests for W5C1 additive attention.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-05/class-01/exercise/test_attention.py -k step1 -q

Runs against the student's exercise file by default. To check the reference
solution, set:  ATTN_FROM=solution
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
    _HERE.parent / "solutions" / "attention.py"
    if os.environ.get("ATTN_FROM") == "solution"
    else _HERE / "attention.py"
)
_spec = importlib.util.spec_from_file_location("attn_under_test", _SRC)
at = importlib.util.module_from_spec(_spec)
sys.modules["attn_under_test"] = at
_spec.loader.exec_module(at)


def test_step1_scores_shape():
    torch.manual_seed(0)
    keys = torch.randn(5, 8)
    q = torch.randn(8)
    W_s = torch.randn(16, 8)
    W_h = torch.randn(16, 8)
    v = torch.randn(16)
    e = at.additive_scores(q, keys, W_s, W_h, v)
    assert e.shape == (5,)


def test_step2_weights_are_a_distribution():
    torch.manual_seed(0)
    m = at.AdditiveAttention(8)
    keys = torch.randn(4, 8)
    context, weights = m(torch.randn(8), keys, keys)
    assert weights.shape == (4,)
    assert torch.all(weights >= 0)
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    assert context.shape == (8,)


def test_step2_context_is_weighted_average_of_values():
    # If one weight dominates, context ~ that value.
    torch.manual_seed(0)
    m = at.AdditiveAttention(8)
    keys = torch.eye(3, 8)
    values = torch.eye(3, 8) * 5.0
    context, weights = m(torch.randn(8), keys, values)
    # context should equal weights @ values exactly
    assert torch.allclose(context, weights @ values, atol=1e-5)


def test_given_heatmap_renders_grid():
    s = at.heatmap(torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                   row_labels=["a", "b"], col_labels=["x", "y"])
    assert isinstance(s, str)
    lines = s.strip().splitlines()
    # header + 2 rows
    assert len(lines) == 3
    # the high-weight cell should use the densest shade '@'
    assert "@" in s
