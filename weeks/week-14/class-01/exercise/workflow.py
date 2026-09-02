#!/usr/bin/env python3
"""W14C1, Router + Workers workflow (STARTER).

You will build a small *agentic workflow*: a router classifies an incoming
request, then dispatches it to a specialized worker. This is the "routing"
pattern from Anthropic's "Building Effective Agents" (2024).

Design goals (mirror real systems):
  * The router returns a label from a FIXED set, with a safe `unknown` fallback.
  * Each worker is its own small function (its own "prompt"/behavior).
  * The whole thing runs against an injectable `llm` callable, so we can pass a
    deterministic MOCK in tests and the real Ollama client in production.

Work through the lab in `README.md`. Each STEP has its own check
(python -m pytest ... -k step1 -q). Or run everything:
    python -m pytest weeks/week-14/class-01/exercise/test_workflow.py -q
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


# An "LLM" here is any callable: prompt (str) -> completion (str).
class LLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


# The labels our router is allowed to emit. Anything else becomes "unknown".
LABELS = ("summarize", "translate", "extract")


@dataclass
class Result:
    """Structured output of the workflow (so downstream code can trust it)."""
    label: str
    output: str
    handled_by: str
    trace: list[str] = field(default_factory=list)


def route(query: str, llm: LLM) -> str:
    """Classify a query into exactly one of LABELS, defensively.

    Args:
        query: whatever the user typed. It is not trusted to be short, clean,
            or on-topic.
        llm: callable taking a prompt and returning the model's raw text. Real
            models pad, capitalize, add punctuation and sometimes refuse, so
            nothing about the reply's shape is guaranteed.

    Returns:
        One of LABELS, or "unknown". Never a raw model string: a label that is
        not in LABELS must become "unknown" rather than flow downstream.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   ask the model to answer with ONE word, listing the labels it may use
    #   take the reply apart: trim it, lower it, and keep only the first word
    #   hand back that word only if it is a label you recognise
    #   anything else is "unknown"
    #
    #   Assume the reply is messy. "  Summarize.\n" and "SUMMARIZE" both have
    #   to come back as "summarize".
    #
    raise NotImplementedError


def worker_summarize(query: str, llm: LLM) -> str:
    """Summarize the user's text in one sentence.

    Args:
        query: the text to summarize.
        llm: callable taking a prompt, returning the model's text.

    Returns:
        The model's reply, unmodified. Workers do not validate; routing already
        decided this is the right worker.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   One line, in the same shape as worker_translate and worker_extract.
    #
    raise NotImplementedError


def worker_translate(query: str, llm: LLM) -> str:
    """Translate the user's text to French."""
    # GIVEN (STEP 2): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return llm(f"Translate the following text into French. Reply with only the translation:\n\n{query}")


def worker_extract(query: str, llm: LLM) -> str:
    """Extract named entities / key items from the text."""
    # GIVEN (STEP 2): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return llm(
        "Extract the key named entities (people, places, organizations, dates) "
        f"from the text as a comma-separated list:\n\n{query}"
    )


def worker_fallback(query: str, llm: LLM) -> str:
    """Safe default when routing fails: never crash, ask for clarification."""
    return "Sorry, I couldn't tell what you wanted. Try: summarize, translate, or extract."


_WORKERS: dict[str, Callable[[str, LLM], str]] = {
    "summarize": worker_summarize,
    "translate": worker_translate,
    "extract": worker_extract,
    "unknown": worker_fallback,
}


def run_workflow(query: str, llm: LLM) -> Result:
    """Route a query to a worker and return a structured Result.

    Args:
        query: the user's request.
        llm: callable taking a prompt, returning text. Both the routing call
            and the worker call go through it.

    Returns:
        A Result carrying the chosen label, the worker's output, which worker
        ran, and a short trace. An unroutable query is not an error: it goes
        to the fallback worker and still returns a Result.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   ask route() which label this query is
    #   look that label up in the _WORKERS table above, defaulting to the
    #       fallback worker when the label is not in it
    #   run the worker on the query
    #   package the label, the output, the worker's name and a short trace
    #       into a Result
    #
    #   The dispatch table already has an "unknown" entry, so the lookup and
    #   the fallback are the same line.
    #
    raise NotImplementedError


# --------------------------------------------------------------------------
# Real-LLM adapter: wraps Ollama into our simple `LLM` callable. Optional.
# --------------------------------------------------------------------------
def make_ollama_llm(model: str = "qwen2.5:0.5b") -> LLM:
    """Return an `LLM` callable backed by a local Ollama model.

    Degrades gracefully: raises a clear RuntimeError if Ollama is unavailable,
    so callers can fall back to the mock.
    """
    import ollama

    client = ollama.Client()

    def _call(prompt: str) -> str:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return resp["message"]["content"].strip()

    return _call


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        # Tiny demo with a real model if available, else a canned mock.
        try:
            llm = make_ollama_llm()
            llm("ping")  # smoke test
            print("Using local Ollama model.\n")
        except Exception as e:  # noqa: BLE001
            print(f"(Ollama unavailable: {e}), using a canned mock LLM.\n")

            def llm(prompt: str) -> str:  # type: ignore[misc]
                if "label" in prompt.lower() or "classify" in prompt.lower():
                    return "summarize"
                return "[mock] response"

        for q in ["Summarize the French Revolution.", "Translate 'hello' for me."]:
            r = run_workflow(q, llm)
            print(f"Q: {q}\n -> [{r.label}] via {r.handled_by}: {r.output}\n")
    except NotImplementedError:
        print("workflow.py is not finished yet: fill in the next TODO in this file, then re-run.")
