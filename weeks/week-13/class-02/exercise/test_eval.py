"""W13C2 tests. Run one step at a time:  pytest -k step3

No LLM here. The strategies take their model as a callable, so a scripted fake
drives them deterministically; that is the same trick that lets you test an
eval harness at all. The real model only appears in run_bench.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "solutions" if os.environ.get("EVAL_FROM") == "solution" else _HERE


def _load(name: str, src: Path):
    for cached in ("tools", "strategies", "eval_suite"):
        sys.modules.pop(cached, None)
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(_HERE))          # data/ and tools.py always live here
    try:
        spec = importlib.util.spec_from_file_location(
            name, (src / f"{name}.py") if (src / f"{name}.py").exists()
            else (_HERE / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(_HERE))
        sys.path.remove(str(src))


S = _load("strategies", _HERE)              # always provided, never a TODO
# No fallback to the reference solution. It used to swap in ../solutions the
# moment a TODO still raised, so a student who had written nothing saw a fully
# green suite. The course sweep verifies the reference with EVAL_FROM=solution
# (scripts/test_all.sh exports every *_FROM var), so nothing needs the fallback.
E = _load("eval_suite", _SRC)


def const_llm(reply: str):
    return lambda _prompt: reply


def R(pid, strategy, correct, calls=1):
    return E.Result(pid=pid, strategy=strategy, predicted=1.0 if correct else 0.0,
                    correct=correct, calls=calls)


# ------------------------------------------------------------- step 1: scoring
def test_step1_is_correct_matches_numerically_not_as_strings():
    assert E.is_correct(18.0, 18)
    assert E.is_correct(1596.00001, 1596)


def test_step1_is_correct_rejects_none_and_wrong():
    assert not E.is_correct(None, 18)
    assert not E.is_correct(17.9, 18)


# -------------------------------------------------------- step 2: one eval run
def test_step2_evaluate_one_scores_and_records_cost():
    p = E.Problem("G01", "What is 6 times 7?", 42.0)
    res = E.evaluate_one(p, "Naive", S.naive, const_llm("The answer is 42"))
    assert res.correct and res.predicted == 42.0
    assert res.calls == 1 and res.pid == "G01" and res.strategy == "Naive"


@pytest.mark.parametrize("name", ["Reflexion+CoT", "Reflexion+ReAct"])
def test_step2_every_reflexion_variant_gets_an_external_evaluator(name):
    # With a model that always answers 41, Reflexion must retry (2 attempts),
    # which only happens if evaluate_one handed it a feedback function. Both
    # variants must qualify, so match on the prefix, not the exact name.
    p = E.Problem("G01", "What is 6 times 7?", 42.0)
    res = E.evaluate_one(p, name, S.STRATEGIES[name],
                         const_llm("Thought: done.\nAction: finish[41]"))
    assert not res.correct
    assert res.attempts == 2, f"{name} should have been given feedback and retried"


def test_step2_reflexion_wraps_its_base_and_costs_more():
    p = E.Problem("G01", "What is 6 times 7?", 42.0)
    llm = const_llm("Thought: done.\nAction: finish[41]")
    base = E.evaluate_one(p, "CoT", S.cot, llm)
    wrapped = E.evaluate_one(p, "Reflexion+CoT", S.STRATEGIES["Reflexion+CoT"], llm)
    assert wrapped.calls > base.calls, "wrapping a base must show up as cost"


# ------------------------------------------------------------- step 3: matrix
def test_step3_run_matrix_covers_every_strategy_and_problem():
    probs = [E.Problem("G01", "q1", 42.0), E.Problem("G02", "q2", 42.0)]
    m = E.run_matrix(probs, {"Naive": S.naive, "CoT": S.cot},
                     const_llm("The answer is 42"))
    assert set(m) == {"Naive", "CoT"}
    assert [r.pid for r in m["Naive"]] == ["G01", "G02"]
    assert all(r.correct for rows in m.values() for r in rows)


# ------------------------------------------------------------ step 4: metrics
def test_step4_success_rate_and_avg_calls():
    rows = [R("G01", "X", True, 1), R("G02", "X", False, 3)]
    assert E.success_rate(rows) == 0.5
    assert E.avg_calls(rows) == 2.0


def test_step4_metrics_handle_an_empty_run():
    assert E.success_rate([]) == 0.0 and E.avg_calls([]) == 0.0


# ------------------------------------------------------- step 5: paired compare
def test_step5_paired_wins_pairs_by_problem_id():
    a = [R("G01", "A", True), R("G02", "A", False), R("G03", "A", True)]
    b = [R("G02", "B", True), R("G01", "B", False), R("G03", "B", True)]  # shuffled
    assert E.paired_wins(a, b) == (1, 1, 1)


# --------------------------------------------------------- step 6: leaderboard
def test_step6_leaderboard_ranks_by_success_then_cost():
    m = {
        "Expensive": [R("G01", "Expensive", True, 8), R("G02", "Expensive", True, 8)],
        "Cheap": [R("G01", "Cheap", True, 1), R("G02", "Cheap", True, 1)],
        "Bad": [R("G01", "Bad", False, 1), R("G02", "Bad", True, 1)],
    }
    ranked = E.leaderboard(m)
    assert [r[0] for r in ranked] == ["Cheap", "Expensive", "Bad"], \
        "ties on success rate must be broken by FEWER calls"
    assert ranked[0][1] == 1.0 and ranked[0][2] == 1.0


def test_step6_format_leaderboard_is_readable():
    m = {"Cheap": [R("G01", "Cheap", True, 1)]}
    out = E.format_leaderboard(m)
    assert "Cheap" in out and "100%" in out


# ------------------------------------------- the provided strategies still work
def test_strategies_parse_a_final_number():
    assert S.last_number("so the answer is 18.0") == 18.0
    assert S.last_number("16 - 3 - 4 = 9, then 9 * 2 = 18") == 18.0
    assert S.last_number("no digits here") is None


def test_strategies_reflexion_pairs_point_at_real_baselines():
    for wrapper, base in S.REFLEXION_PAIRS:
        assert wrapper in S.STRATEGIES and base in S.STRATEGIES


def test_strategies_reflexion_stops_early_when_the_first_try_is_right():
    llm = const_llm("Thought: done.\nAction: finish[42]")
    run = S.STRATEGIES["Reflexion+ReAct"]("q", llm,
                                          feedback_fn=lambda a: (a == 42.0, "ok"))
    assert run.attempts == 1, "a correct first attempt must not pay for a retry"


def test_strategies_react_uses_the_calculator():
    replies = iter(["Thought: multiply.\nAction: calc[6 * 7]",
                    "Thought: done.\nAction: finish[42]"])
    run = S.react("What is 6 times 7?", lambda _p: next(replies))
    assert run.answer == 42.0 and run.steps == 1 and run.calls == 2


def test_strategies_react_survives_a_malformed_reply():
    replies = iter(["I do not know what to do.",
                    "Thought: ok.\nAction: finish[42]"])
    run = S.react("q", lambda _p: next(replies))
    assert run.answer == 42.0
