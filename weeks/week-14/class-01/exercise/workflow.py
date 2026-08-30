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
    """Ask the LLM to classify `query` into exactly one LABEL.

    The LLM is prompted to reply with a single word. We must DEFEND against
    messy output: lowercase, strip, and map anything not in LABELS to "unknown".

    STEP 1 (check with: pytest -k step1):
      1. Build a prompt that lists the allowed labels and asks for ONE word.
      2. Call `llm(prompt)`.
      3. Normalize the reply (strip/lower, take the first word/token).
      4. Return it if in LABELS, else return "unknown".
    """
    raise NotImplementedError("Implement route()")


def worker_summarize(query: str, llm: LLM) -> str:
    """Summarize the user's text in one sentence."""
    # TODO (STEP 2): prompt the llm to summarize `query` in one sentence.
    #                Check with: pytest -k step2
    raise NotImplementedError("Implement worker_summarize()")


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


def run_workflow(query: str, llm: LLM) -> Result:
    """Orchestrate: route -> dispatch to a worker -> return a structured Result.

    STEP 4 (check with: pytest -k step4):
      1. Call route() to get a label.
      2. Pick the matching worker from a dispatch table (fallback for "unknown").
      3. Call the worker, build and return a `Result` with a small trace.
    """
    raise NotImplementedError("Implement run_workflow()")


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
