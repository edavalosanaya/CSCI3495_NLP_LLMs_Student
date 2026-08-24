"""Tests for W11C1 json_lab (offline-testable parts).

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-11/class-01/exercise/test_json_lab.py -k step1 -q

Set JSON_LAB_FROM=solution to test the reference solution.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = (
    _HERE.parent / "solutions" / "json_lab.py"
    if os.environ.get("JSON_LAB_FROM") == "solution"
    else _HERE / "json_lab.py"
)
_spec = importlib.util.spec_from_file_location("json_lab_under_test", _SRC)
jl = importlib.util.module_from_spec(_spec)
sys.modules["json_lab_under_test"] = jl
_spec.loader.exec_module(jl)


def _implemented(fn, *args):
    try:
        fn(*args)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True  # raised a real error -> it's implemented


pytestmark = pytest.mark.skipif(
    not _implemented(jl.extract_json, '{"a": 1}'),
    reason="json_lab not implemented yet (fill in the TODOs)",
)


def test_step1_extract_plain():
    assert jl.extract_json('{"name": "Mug", "price": 5.0}') == {"name": "Mug", "price": 5.0}


def test_step1_extract_with_fence_and_prose():
    text = 'Here you go:\n```json\n{"name": "Mug", "price": 5}\n```\nThanks!'
    assert jl.extract_json(text) == {"name": "Mug", "price": 5}


def test_step1_extract_no_json_raises():
    with pytest.raises(ValueError):
        jl.extract_json("there is no object here")


def test_step2_validate_ok():
    if not _implemented(jl.validate, {}, jl.SCHEMA):
        pytest.skip("validate not implemented")
    rec = {"name": "Mug", "price": 5.0, "in_stock": True, "rating": 4}
    assert jl.validate(rec, jl.SCHEMA) == []


def test_step2_validate_catches_errors():
    if not _implemented(jl.validate, {}, jl.SCHEMA):
        pytest.skip("validate not implemented")
    bad = {"name": "Mug", "price": "five", "in_stock": True, "rating": 9}
    errs = jl.validate(bad, jl.SCHEMA)
    assert any("price" in e for e in errs)   # wrong type
    assert any("rating" in e for e in errs)  # out of range


def test_step2_validate_missing_field():
    if not _implemented(jl.validate, {}, jl.SCHEMA):
        pytest.skip("validate not implemented")
    errs = jl.validate({"name": "Mug"}, jl.SCHEMA)
    assert len(errs) >= 1


def test_step2_validate_bool_not_int_for_rating():
    """bool is a subclass of int in Python; rating=True must NOT pass as int 1."""
    if not _implemented(jl.validate, {}, jl.SCHEMA):
        pytest.skip("validate not implemented")
    rec = {"name": "Mug", "price": 1.0, "in_stock": True, "rating": True}
    assert any("rating" in e for e in jl.validate(rec, jl.SCHEMA))


def test_step3_generate_valid_retries_to_success():
    if not _implemented(jl.validate, {}, jl.SCHEMA):
        pytest.skip("validate not implemented")
    if not _implemented(jl.generate_valid, jl.StubModel(), jl.SCHEMA):
        pytest.skip("generate_valid not implemented")
    m = jl.StubModel()
    rec = jl.generate_valid(m, jl.SCHEMA, max_retries=3)
    assert jl.validate(rec, jl.SCHEMA) == []
    assert m.calls >= 3  # stub is invalid on attempts 1 and 2
