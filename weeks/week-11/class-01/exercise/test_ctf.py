"""Deterministic tests for the W11C1 Injection CTF.

Test names carry their lab step, so you can check one step at a time:
    python -m pytest weeks/week-11/class-01/exercise/test_ctf.py -k step1 -q

These run fully offline (the model is a rule-based stand-in). They check:
  - PHASE 1: the student's attacks actually break the UNGUARDED assistant.
  - PHASE 2: the student's defenses block those SAME attacks on the GUARDED one,
    while a human-approved privileged action still succeeds.

Run:  python -m pytest weeks/week-11/class-01/exercise/test_ctf.py -q
Tests prefer the reference solution if the starter TODOs are unimplemented, so the
suite is green out of the box and turns red only on a real regression.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SOLUTION = HERE.parent / "solutions" / "ctf.py"


def _load():
    """Load the student's ctf.py; if its TODOs raise, fall back to the solution."""
    spec = importlib.util.spec_from_file_location("ctf_student", HERE / "ctf.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ctf_student"] = mod
    spec.loader.exec_module(mod)
    try:
        mod.attack_leak_secret()
        mod.guard_output("x")
        mod.tool_allowed("lookup_order", False)
        return mod
    except NotImplementedError:
        spec2 = importlib.util.spec_from_file_location("ctf_ref", SOLUTION)
        ref = importlib.util.module_from_spec(spec2)
        sys.modules["ctf_ref"] = ref  # needed so @dataclass can resolve __module__
        spec2.loader.exec_module(ref)
        return ref


ctf = _load()


# --------------------------- PHASE 1: attacks work ---------------------------
def test_step1_attack_leak_breaks_unguarded():
    msg = ctf.attack_leak_secret()
    reply = ctf.naive_model(ctf.SYSTEM_PROMPT, msg)
    assert ctf.SECRET_TOKEN in reply.text, "leak attack should expose the secret on the naive model"


def test_step2_attack_refund_breaks_unguarded():
    msg = ctf.attack_force_refund()
    reply = ctf.naive_model(ctf.SYSTEM_PROMPT, msg)
    assert reply.tool == "refund", "refund attack should fire the privileged tool on the naive model"
    assert reply.tool_arg == "1001"


# --------------------------- PHASE 2: defenses block -------------------------
def test_step3_guarded_blocks_leak():
    reply = ctf.guarded_assistant(ctf.attack_leak_secret())
    assert ctf.SECRET_TOKEN not in reply.text, "guarded assistant must not leak the secret"
    assert any(f.startswith("injection") or f == "leak:secret" for f in reply.blocked)


def test_step5_guarded_blocks_refund():
    reply = ctf.guarded_assistant(ctf.attack_force_refund())
    assert reply.tool is None, "privileged refund must be blocked without approval"
    assert any("tool-blocked" in f or "injection" in f for f in reply.blocked)


def test_step5_approved_refund_succeeds():
    reply = ctf.guarded_assistant("refund order 1001", approved=True)
    assert "1001" in reply.text and "Refund processed" in reply.text


def test_step6_normal_lookup_still_works():
    reply = ctf.guarded_assistant("Can you check order 1001?")
    assert "Blue Mug" in reply.text
    assert reply.blocked == []


# --------------------------- unit: the guards --------------------------------
def test_step4_guard_output_redacts_secret():
    text, flags = ctf.guard_output(f"the token is {ctf.SECRET_TOKEN}")
    assert ctf.SECRET_TOKEN not in text
    assert "leak:secret" in flags


def test_step5_tool_allowlist():
    assert ctf.tool_allowed("lookup_order", False) is True
    assert ctf.tool_allowed("refund", False) is False        # privileged, no approval
    assert ctf.tool_allowed("refund", True) is True          # human approved
    assert ctf.tool_allowed("rm_rf", True) is False          # unknown tool denied
    assert ctf.tool_allowed(None, False) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
