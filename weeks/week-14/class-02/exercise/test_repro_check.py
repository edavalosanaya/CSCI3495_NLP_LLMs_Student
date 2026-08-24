"""Tests for the W14C2 reproducibility quick-check helper (offline, fast)."""
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("repro_check", _HERE / "repro_check.py")
rc = importlib.util.module_from_spec(_spec)
sys.modules["repro_check"] = rc
_spec.loader.exec_module(rc)


def test_parses_clean_source():
    assert rc.parses_cleanly("x = 1\nprint(x)\n") is True


def test_parses_rejects_syntax_error():
    assert rc.parses_cleanly("def f(:\n  pass") is False


def test_find_seed_hints_detects_seeding():
    src = "import numpy as np\nnp.random.seed(0)\nimport torch\ntorch.manual_seed(0)"
    hints = rc.find_seed_hints(src)
    assert "np.random.seed" in hints
    assert "torch.manual_seed" in hints


def test_find_seed_hints_empty_when_absent():
    assert rc.find_seed_hints("print('hello')") == []


def test_check_file_missing(tmp_path):
    report = rc.check_file(tmp_path / "nope.py")
    assert report["ok"] is False
    assert "not found" in report["reason"]


def test_check_file_clean_and_seeded(tmp_path):
    f = tmp_path / "good.py"
    f.write_text("import random\nrandom.seed(7)\nprint('ok')\n")
    report = rc.check_file(f)
    assert report["ok"] is True
    assert report["deterministic_hint"] is True
    assert "random.seed" in report["seed_hints"]


def test_check_file_clean_but_unseeded(tmp_path):
    f = tmp_path / "nosseed.py"
    f.write_text("print('no seeds here')\n")
    report = rc.check_file(f)
    assert report["ok"] is True
    assert report["deterministic_hint"] is False


def test_format_report_warns_on_missing_seed(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("print(1)\n")
    out = rc.format_report(f, rc.check_file(f))
    assert "WARN" in out and "seeding" in out


def test_main_returns_zero_on_good_file(tmp_path):
    f = tmp_path / "g.py"
    f.write_text("import random\nrandom.seed(1)\n")
    assert rc.main([str(f)]) == 0


def test_main_returns_nonzero_on_missing():
    assert rc.main(["/no/such/file/here.py"]) == 1
