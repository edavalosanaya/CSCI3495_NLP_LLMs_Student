"""HW6 starter, Building an LLM Agent with Tool Use (ReAct).

You will build a small **ReAct** agent: a loop where a language model alternates
between *reasoning* (Thought) and *acting* (calling a tool), observing the
result, and continuing until it produces a Final Answer.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language
Models" (2022), arXiv:2210.03629.

The two tools are entirely local and safe:
  * ``calculator``, evaluates a basic arithmetic expression (no eval()).
  * ``search``, keyword lookup over a small fixed corpus.

The tool dispatch and the ReAct loop are unit-testable with a MOCK LLM that
returns canned Thought/Action text. The real Ollama path degrades gracefully.

Run the tests with:
    docker compose -f docker/docker-compose.yml run --rm course \
        python -m pytest homeworks/hw6/tests -q
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Task 1, Safe tools.
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result as a string.

    Support + - * / ( ) and decimal numbers. You MUST NOT use ``eval``/``exec``.
    Implement a small shunting-yard / recursive-descent parser, or tokenize and
    use Python's ``ast`` module restricted to arithmetic nodes.
    On a malformed expression or division by zero, return a string starting with
    "Error".
    """
    # TODO: implement a SAFE evaluator (no eval/exec).
    raise NotImplementedError


# A tiny fixed corpus the `search` tool queries. (No network.)
CORPUS = {
    "paris": "Paris is the capital of France. Its population is about 2.1 million.",
    "france": "France is a country in Western Europe. Its capital is Paris.",
    "python": "Python is a programming language created by Guido van Rossum in 1991.",
    "transformer": "The Transformer architecture was introduced in 'Attention Is All You Need' (2017).",
    "react": "ReAct (Yao et al., 2022) interleaves reasoning traces and actions in language models.",
}


def search(query: str) -> str:
    """Return the corpus entry whose key best matches the query.

    Match = the corpus key with the most query-word overlap (case-insensitive);
    a key also counts as matched if it appears as a substring of the query.
    If nothing matches, return "No results found.".
    """
    # TODO: implement keyword search over CORPUS.
    raise NotImplementedError


# Registry mapping tool name -> callable(str) -> str.
TOOLS = {"calculator": calculator, "search": search}


# ---------------------------------------------------------------------------
# Task 2, Parse a ReAct step from model text.
# ---------------------------------------------------------------------------
@dataclass
class Step:
    """One parsed ReAct step.

    Exactly one of (action, final_answer) is meaningful:
      * if ``final_answer`` is not None -> the agent is done.
      * else ``action`` is the tool name and ``action_input`` its argument.
    ``thought`` holds the reasoning text (may be "").
    """

    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    final_answer: str | None = None


def parse_step(text: str) -> Step:
    """Parse a model turn in the ReAct format.

    The expected format (case-insensitive labels) is some of:
        Thought: <reasoning>
        Action: <tool name>
        Action Input: <argument>
    OR a terminal line:
        Final Answer: <answer>

    Rules:
      * If a "Final Answer:" line is present, return a Step with that
        final_answer (and the thought if any); ignore any Action.
      * Otherwise parse Thought / Action / Action Input. ``action`` and
        ``action_input`` are stripped strings (action_input may be "").
      * Labels may be surrounded by extra whitespace/newlines.
    """
    # TODO: implement robust parsing (regex per label is fine).
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 3, Run a tool and format the observation.
# ---------------------------------------------------------------------------
def run_tool(step: Step) -> str:
    """Dispatch ``step.action`` to the matching tool in TOOLS with
    ``step.action_input`` and return its string result. If the tool name is
    unknown, return a string starting with "Error: unknown tool".
    """
    # TODO: implement dispatch via the TOOLS registry.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 4, The ReAct loop.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a ReAct agent. Solve the question using tools.
On each turn respond in this exact format:

Thought: <your reasoning>
Action: <one of: calculator, search>
Action Input: <the argument for the tool>

After you see the Observation, continue. When you can answer, respond:

Thought: <reasoning>
Final Answer: <the answer>

Available tools:
- calculator: evaluate an arithmetic expression, e.g. "12 * (3 + 4)".
- search: look up a fact in the local knowledge base, e.g. "capital of France".
"""


def build_react_prompt(question: str, history: list[str]) -> str:
    """Assemble the prompt: the system instructions, the Question, and the
    running transcript of prior Thought/Action/Observation lines (``history``).
    Return a single string ending in a cue for the model's next turn.
    """
    # TODO: implement
    raise NotImplementedError


def react_loop(question: str, llm, max_steps: int = 5) -> dict:
    """Run the ReAct loop.

    ``llm`` is a callable ``llm(prompt: str) -> str`` returning the model's next
    turn (a mock in tests; an Ollama wrapper in production).

    Loop up to ``max_steps`` times:
      * build the prompt from the question + history,
      * call the llm, parse the step,
      * if it has a final_answer -> stop and return it,
      * else run the tool, append "Thought/Action/Action Input/Observation" to
        the history, and continue.

    Return a dict:
      {'answer': <str or None if no final answer within max_steps>,
       'steps': <number of llm calls made>,
       'history': <the transcript list>}
    """
    # TODO: implement the loop.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Production LLM wrapper (used outside tests; needs Ollama running).
# ---------------------------------------------------------------------------
def ollama_llm(model: str = "qwen2.5:0.5b"):
    """Return an ``llm(prompt) -> str`` callable backed by a local Ollama model.

    Stops generation at "Observation:" so the model doesn't hallucinate tool
    output. Requires a running Ollama server with ``model`` pulled.
    """

    def _call(prompt: str) -> str:
        import ollama

        resp = ollama.generate(
            model=model, prompt=prompt, options={"stop": ["Observation:"]}
        )
        return resp["response"]

    return _call
