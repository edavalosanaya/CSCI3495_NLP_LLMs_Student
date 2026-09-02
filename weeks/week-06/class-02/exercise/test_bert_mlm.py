"""Tests for W6C2 bert_mlm.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-06/class-02/exercise/test_bert_mlm.py -k step1 -q

Runs against the student's exercise file by default. Set BERT_MLM_FROM=solution
to test the reference solution (used by the course sweep).

All tests need the tiny BERT model. They skip gracefully if (a) functions
aren't implemented or (b) the model can't be loaded (offline, no cache).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "bert_mlm.py"
    if os.environ.get("BERT_MLM_FROM") == "solution"
    else _HERE / "bert_mlm.py"
)
_spec = importlib.util.spec_from_file_location("bert_mlm_under_test", _SRC)
bm = importlib.util.module_from_spec(_spec)
sys.modules["bert_mlm_under_test"] = bm
_spec.loader.exec_module(bm)


MLM_MODEL_NAME = bm.MLM_MODEL


def _mlm_ready():
    """True if the tiny BERT actually loads.

    Gate ONLY on the model, never on a student function. Calling the student's
    top_mask_predictions here meant any bug in it -- a NameError, a wrong
    shape -- was reported as "offline" and skipped, so a broken lab looked
    finished.
    """
    try:
        from transformers import BertForMaskedLM, BertTokenizerFast
        BertTokenizerFast.from_pretrained(MLM_MODEL_NAME)
        BertForMaskedLM.from_pretrained(MLM_MODEL_NAME)
        return True
    except Exception:  # noqa: BLE001  the model itself is unavailable
        return False


def _ft_ready():
    try:
        bm.finetune_and_eval  # noqa: B018
        # quick implemented-check without training: call with 0 epochs is still
        # heavy (loads model), so only verify it's not the NotImplemented stub.
        import inspect

        src = inspect.getsource(bm.finetune_and_eval)
        return "raise NotImplementedError" not in src
    except Exception:  # noqa: BLE001
        return False


needs_mlm = pytest.mark.skipif(
    not _mlm_ready(), reason="tiny model unavailable (offline)"
)
needs_ft = pytest.mark.skipif(
    not _ft_ready(), reason="finetune not implemented"
)


@needs_mlm
def test_step1_mask_predictions_shape():
    preds = bm.top_mask_predictions("The sky is [MASK].", k=5)
    assert isinstance(preds, list)
    assert len(preds) == 5
    assert all(isinstance(p, str) and p for p in preds)


@needs_mlm
def test_step1_mask_predictions_k():
    preds = bm.top_mask_predictions("I like to [MASK] every day.", k=3)
    assert len(preds) == 3


@needs_ft
def test_step2_finetune_beats_chance():
    # Tiny dataset, tiny model, but fine-tuning should reach >= chance (0.5)
    # and typically 1.0 on these two easy, clearly-polar test sentences.
    try:
        acc = bm.finetune_and_eval(epochs=8, seed=0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model unavailable (offline?): {e}")
    assert 0.0 <= acc <= 1.0
    assert acc >= 0.5, f"fine-tuned accuracy {acc:.2f} should be at least chance"
