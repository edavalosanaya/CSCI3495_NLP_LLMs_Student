"""W13 agent tools, provided (same safe, network-free tools as W12).

  * calc[expr]    -> SAFE arithmetic (AST whitelist, no eval)
  * search[query] -> keyword search over a FIXED local corpus
"""
from __future__ import annotations

import ast
import operator
import re

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


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "Error: empty expression"
    try:
        result = _eval_node(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:  # noqa: BLE001
        return f"Error: could not evaluate '{expr}'"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


CORPUS: dict[str, str] = {
    "react": (
        "ReAct (Yao et al., 2022) interleaves reasoning traces (Thought) with "
        "actions (tool calls), letting a model plan, act, and update on observations."
    ),
    "reflexion": (
        "Reflexion (Shinn et al., 2023) has an agent verbally reflect on failures "
        "and store the lesson in memory to improve on the next attempt."
    ),
    "generative agents": (
        "Generative Agents (Park et al., 2023) gave 25 agents a memory stream with "
        "retrieval, reflection, and planning to simulate believable behavior."
    ),
    "transformer": (
        "The Transformer (Vaswani et al., 2017) replaced recurrence with "
        "self-attention and is the backbone of modern LLMs."
    ),
    "paris": "Paris is the capital of France; its population is about 2.1 million.",
    "tokyo": "Tokyo is the capital of Japan; its population is about 14 million.",
    "rome": "Rome is the capital of Italy; it was founded, by tradition, in 753 BC.",
    "speed of light": "The speed of light is about 299,792 kilometers per second.",
    "everest": "Mount Everest is 8,849 meters tall, the highest mountain on Earth.",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def search(query: str) -> str:
    query = query.strip()
    if not query:
        return "Error: empty query"
    q = _tokens(query)
    best_key, best_score = None, 0
    for key, text in CORPUS.items():
        score = len(q & _tokens(key + " " + text))
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return f"No results found for '{query}'."
    return CORPUS[best_key]


TOOLS = {"calc": calculator, "search": search}
