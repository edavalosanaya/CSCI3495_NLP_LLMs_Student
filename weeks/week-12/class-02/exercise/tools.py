"""W12C2 lab, the agent's tools (starter).

Four safe, deterministic, network-free tools:

  * calc[expr]         a SAFE calculator (parses to an AST, never calls eval)
  * today[]            the real current date
  * weather[city, day] a high temperature from a local stand-in service
  * search[query]      keyword search over a small FIXED local corpus

Every tool takes ONE string and returns ONE string Observation. Tools never
raise to the caller: an error becomes a readable Observation so the agent can
read it and recover. That contract is what makes the loop robust.

The weather series is keyed by OFFSET FROM TODAY (0 = today, 1 = yesterday),
not by absolute date, so this lab produces the same numbers whatever day you
run it. A real deployment would call an API here; the shape of the tool, one
string in and one string out, would be identical.
"""
from __future__ import annotations

import ast
import datetime as _dt
import math
import operator
import re

# --------------------------------------------------------------------------
# Safe calculator
# --------------------------------------------------------------------------
# We parse the expression to an AST and evaluate only a whitelist of nodes.
# The key safety idea: NEVER call eval() on model-produced text.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Functions the model is allowed to call inside an expression.
_FUNCS = {
    "log": math.log, "log10": math.log10, "sqrt": math.sqrt, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):  # numbers only
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("unknown function")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expr: str) -> str:
    """Evaluate an arithmetic expression safely. Returns an Observation string.

    `^` is rewritten to `**` first. This matters more than it looks: in Python
    `^` is bitwise XOR, so `log(3^2 * 16 - 10)` would silently evaluate as
    log(3 XOR 22) = log(21) and return a plausible, wrong number. A tool that
    is quietly wrong is worse than one that errors.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    # 1. strip the expression and rewrite "^" as "**"
    # 2. empty -> "Error: empty expression"
    # 3. _eval_node(ast.parse(expr, mode="eval")) inside try/except
    #    ZeroDivisionError -> "Error: division by zero"
    #    anything else     -> f"Error: could not evaluate '{expr}'"
    # 4. render a whole float as an int, then return str(result)
    raise NotImplementedError


# --------------------------------------------------------------------------
# Today's date
# --------------------------------------------------------------------------
def today(_arg: str = "") -> str:
    """Return the real current date as YYYY-MM-DD. Ignores its argument.

    This is the cheapest possible demonstration that a tool beats memory: the
    date is not in the weights, and no amount of prompting can put it there.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


# --------------------------------------------------------------------------
# Weather (a local stand-in for a real API)
# --------------------------------------------------------------------------
# index 0 = today, 1 = yesterday, 2 = two days ago, ...
_SERIES: dict[str, list[float]] = {
    "san antonio": [101.0, 94.0, 97.0, 99.0, 96.0, 93.0, 95.0],
    "austin":      [99.0, 96.0, 95.0, 98.0, 94.0, 92.0, 93.0],
    "boston":      [78.0, 81.0, 76.0, 74.0, 79.0, 77.0, 80.0],
    "seattle":     [69.0, 72.0, 68.0, 70.0, 71.0, 73.0, 67.0],
}


def _day_offset(day: str) -> int | None:
    """Turn 'today' / 'yesterday' / 'YYYY-MM-DD' into days-before-today."""
    day = day.strip().strip('"\'').lower().replace("_", " ")
    if day in ("", "today", "now"):
        return 0
    if day == "yesterday":
        return 1
    try:
        d = _dt.date.fromisoformat(day)
    except ValueError:
        return None
    return (_dt.date.today() - d).days


def weather(arg: str) -> str:
    """`weather[city, day]` -> the high temperature in F, as a bare number.

    Accepts a sloppy city ("San_Antonio") and a relative day ("yesterday"),
    because a tool that only accepts one exact spelling turns every small
    model into a loop. Being forgiving about the INPUT while staying exact
    about the OUTPUT is most of good tool design.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    # split on the first comma; normalise the city (strip, quotes, lowercase,
    # "_" -> " "); reject an unknown city; use _day_offset(day) for the date;
    # reject an offset outside the series. Return str(temperature).
    raise NotImplementedError


# --------------------------------------------------------------------------
# Local search over a FIXED corpus (no network)
# --------------------------------------------------------------------------
CORPUS: dict[str, str] = {
    "react": (
        "ReAct (Yao et al., 2022) interleaves reasoning traces (Thought) with "
        "actions (tool calls), letting a model plan, act, and update on observations."
    ),
    "toolformer": (
        "Toolformer (Schick et al., 2023) teaches a model, via self-supervision, "
        "to decide which API to call, when, and with what arguments."
    ),
    "reflexion": (
        "Reflexion (Shinn et al., 2023) has an agent verbally reflect on failures "
        "and store the lesson in memory to improve on the next attempt."
    ),
    "transformer": (
        "The Transformer (Vaswani et al., 2017) replaced recurrence with "
        "self-attention and is the backbone of modern LLMs."
    ),
    "ollama": (
        "Ollama runs open-weight LLMs locally with no API key, used throughout "
        "this course for free, private, CPU-friendly inference."
    ),
    "python": (
        "Python is the primary language of this course; agents here are plain "
        "Python that parse model output and dispatch to tools."
    ),
    "capital of france": "The capital of France is Paris.",
    "speed of light": "The speed of light is about 299,792 kilometers per second.",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def search(query: str) -> str:
    """Return the best-matching corpus entry by word overlap. Network-free."""
    # TODO (STEP 4): implement. Check with: pytest -k step4
    # score every CORPUS entry by word overlap with the query and return the
    # best entry's text; no overlap -> f"No results found for '{query}'."
    raise NotImplementedError


# The registry the agent dispatches against: tool name -> callable(str) -> str.
# The lab builds this up one entry at a time, which is the whole point: each
# new key is a new thing the model stops having to guess about.
# TODO (STEP 5): register your tools here. Check with: pytest -k step5
# The prompt is BUILT FROM THIS DICT, so a tool you do not register is a tool
# the model is never told about and can never call.
TOOLS = {}
