#!/usr/bin/env python3
"""W10C1 starter, a tiny prompt-experiment harness."""
from __future__ import annotations
import os
import re
from typing import Callable

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

# A tiny labeled sentiment set (review -> "positive" / "negative").
DATASET: list[tuple[str, str]] = [
    ("Absolutely loved every minute of it.", "positive"),
    ("A complete waste of two hours.", "negative"),
    ("The acting was superb and moving.", "positive"),
    ("Boring, predictable, and far too long.", "negative"),
    ("One of the best films this year.", "positive"),
    ("I want my money back.", "negative"),
    ("Charming, funny, and heartfelt.", "positive"),
    ("The plot made no sense at all.", "negative"),
]

DEMOS: list[tuple[str, str]] = [
    ("What a delightful surprise!", "positive"),
    ("Dull and forgettable.", "negative"),
]


# ----------------------------- TODO 1 -----------------------------
def accuracy(preds: list[str], golds: list[str]) -> float:
    """Exact-match accuracy in [0, 1]. Assumes len(preds) == len(golds) > 0."""
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(golds)


# ----------------------------- TODO 2 -----------------------------
def build_fewshot_prompt(instruction: str, demos: list[tuple[str, str]], query: str) -> str:
    """Build the four-part few-shot prompt: instruction, demos, query, cue.

    Args:
        instruction: the task description, first line of the prompt.
        demos: (text, label) pairs to show as worked examples. May be empty,
            which makes this a zero-shot prompt.
        query: the item the model is actually being asked about.

    Returns:
        One prompt string ending in a BARE output cue with no label after it.
        That dangling cue is what tells the model the shape to answer in, and
        leaving it off is the usual bug.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The shape is drawn in README section 2. Build one string.
    #
    #   start with the instruction, then a blank line
    #   for every demonstration, add its review line, its sentiment line, and
    #       a blank line after it
    #   finish with the query's review line and a BARE "Sentiment:" line
    #   join it all with newlines
    #
    #   That last line has no label after it. It is the cue the model is meant
    #   to complete, and leaving it off is the usual bug.
    #
    raise NotImplementedError


def build_zeroshot_prompt(instruction: str, demos, query: str) -> str:
    """A zero-shot prompt ignores demos entirely (provided for comparison)."""
    return f"{instruction}\n\nReview: {query}\nSentiment:"


# ----------------------------- TODO 3 -----------------------------
def run_experiment(model: "Model", prompt_fn: Callable, dataset, demos) -> tuple[list[str], float]:
    """Run one prompting strategy over the whole dataset and score it.

    Args:
        model: anything with .generate(prompt) -> str. The stub model keeps
            this runnable with no LLM at all.
        prompt_fn: called as prompt_fn(instruction, demos, query). Swapping
            this is how the zero-shot and few-shot variants are compared.
        dataset: (text, gold label) pairs.
        demos: the demonstrations to embed, ignored by the zero-shot builder.

    Returns:
        (predictions, accuracy). Predictions are parsed labels, one per item,
        in dataset order.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   for each item and its gold label:
    #       build the prompt with prompt_fn, using the INSTRUCTION constant
    #       ask the model to generate from it
    #       turn the raw reply into a label with the given parse_label
    #   score the predictions against the gold labels with the given accuracy
    #   return both
    #
    raise NotImplementedError


INSTRUCTION = "Classify the movie review's sentiment as positive or negative."


def parse_label(text: str) -> str:
    """Map free-form model text to 'positive'/'negative' (first signal wins)."""
    t = text.lower()
    pos = re.search(r"positive|good|favorable", t)
    neg = re.search(r"negative|bad|unfavorable", t)
    if pos and (not neg or pos.start() < neg.start()):
        return "positive"
    if neg:
        return "negative"
    return "unknown"


# --------------------------- model backends ---------------------------
class Model:
    """Interface: .generate(prompt) -> str, deterministic."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class StubModel(Model):
    """Deterministic keyword classifier, no network. Lets the harness run offline.

    It is intentionally imperfect (it only knows a few words) so prompt quality
    still matters a little, but it always returns a parseable label.
    """

    POS = {"loved", "superb", "best", "charming", "funny", "heartfelt", "delightful", "moving"}
    NEG = {"waste", "boring", "predictable", "money", "back", "sense", "dull", "forgettable"}

    def generate(self, prompt: str) -> str:
        # Look only at the final 'Review:' line (the query) for fairness.
        lines = [ln for ln in prompt.splitlines() if ln.strip().lower().startswith("review:")]
        query = lines[-1] if lines else prompt
        words = set(re.findall(r"[a-z']+", query.lower()))
        score = len(words & self.POS) - len(words & self.NEG)
        return "positive" if score >= 0 else "negative"


class OllamaModel(Model):
    """Real local model via Ollama, temperature 0 for reproducibility."""

    def __init__(self, name: str = MODEL):
        import ollama  # imported lazily so the stub path needs no dependency
        self.client = ollama.Client()
        self.name = name

    def generate(self, prompt: str) -> str:
        resp = self.client.chat(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "seed": 0, "num_predict": 8},
        )
        return resp["message"]["content"]


def get_model() -> Model:
    """Try the real local model; fall back to the offline stub with a message."""
    try:
        m = OllamaModel()
        m.generate("Review: test\nSentiment:")  # probe
        print(f"[model] using Ollama model '{MODEL}'")
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[model] Ollama unavailable ({type(e).__name__}); using offline stub.")
        print("        Start it with: docker compose ... up -d ollama && ollama pull " + MODEL)
        return StubModel()


def main() -> int:
    model = get_model()

    experiments = {
        "zero-shot": build_zeroshot_prompt,
        "few-shot": build_fewshot_prompt,
    }
    print(f"\n{'variant':<14}{'accuracy':>10}")
    print("-" * 24)
    for name, fn in experiments.items():
        try:
            _, acc = run_experiment(model, fn, DATASET, DEMOS)
        except NotImplementedError:
            print(f"{name:<14}{'(TODO)':>10}")
            continue
        print(f"{name:<14}{acc:>9.0%}")

    # TODO (stretch): add your own prompt variant here and compare. Prompt golf:
    #   shortest prompt that holds the best score wins.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
