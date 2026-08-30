#!/usr/bin/env python3
"""W14C1, Router + Workers workflow (REFERENCE SOLUTION)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class LLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


LABELS = ("summarize", "translate", "extract")


@dataclass
class Result:
    label: str
    output: str
    handled_by: str
    trace: list[str] = field(default_factory=list)


def route(query: str, llm: LLM) -> str:
    prompt = (
        "Classify the user's request into exactly one label from this set: "
        f"{', '.join(LABELS)}.\n"
        "Reply with ONLY the single label word, nothing else.\n\n"
        f"Request: {query}\nLabel:"
    )
    raw = llm(prompt)
    # Defend against messy output: lowercase, strip punctuation, take first token.
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    token = token.strip(".,:;!?\"'`")
    return token if token in LABELS else "unknown"


def worker_summarize(query: str, llm: LLM) -> str:
    return llm(f"Summarize the following in one clear sentence:\n\n{query}")


def worker_translate(query: str, llm: LLM) -> str:
    return llm(f"Translate the following text into French. Reply with only the translation:\n\n{query}")


def worker_extract(query: str, llm: LLM) -> str:
    return llm(
        "Extract the key named entities (people, places, organizations, dates) "
        f"from the text as a comma-separated list:\n\n{query}"
    )


def worker_fallback(query: str, llm: LLM) -> str:
    return "Sorry, I couldn't tell what you wanted. Try: summarize, translate, or extract."


_WORKERS: dict[str, Callable[[str, LLM], str]] = {
    "summarize": worker_summarize,
    "translate": worker_translate,
    "extract": worker_extract,
    "unknown": worker_fallback,
}


def run_workflow(query: str, llm: LLM) -> Result:
    label = route(query, llm)
    worker = _WORKERS.get(label, worker_fallback)
    output = worker(query, llm)
    handled_by = worker.__name__
    trace = [f"route -> {label}", f"dispatch -> {handled_by}"]
    return Result(label=label, output=output, handled_by=handled_by, trace=trace)


def make_ollama_llm(model: str = "qwen2.5:0.5b") -> LLM:
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
    try:
        llm = make_ollama_llm()
        llm("ping")
        print("Using local Ollama model.\n")
    except Exception as e:  # noqa: BLE001
        print(f"(Ollama unavailable: {e}), using a canned mock LLM.\n")

        def llm(prompt: str) -> str:  # type: ignore[misc]
            if "Label:" in prompt:
                return "summarize"
            return "[mock] response"

    for q in ["Summarize the French Revolution.", "Translate 'hello' for me."]:
        r = run_workflow(q, llm)
        print(f"Q: {q}\n -> [{r.label}] via {r.handled_by}: {r.output}\n")
