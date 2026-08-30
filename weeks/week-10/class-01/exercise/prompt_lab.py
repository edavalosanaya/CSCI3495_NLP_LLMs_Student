#!/usr/bin/env python3
"""W10C1 starter, a tiny prompt-experiment harness.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-10/class-01/exercise/test_prompt_lab.py -k step1 -q

When every step is done, run the full experiment (real local model if
available, else a stub):
    python weeks/week-10/class-01/exercise/prompt_lab.py

Design notes:
- The LLM call degrades gracefully: if Ollama / the model is missing we fall
  back to a deterministic keyword stub so the pipeline + metric still run.
- Hold decoding fixed (temperature 0) so score differences are due to the PROMPT.
"""
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
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(golds)


# ----------------------------- TODO 2 -----------------------------
def build_fewshot_prompt(instruction: str, demos: list[tuple[str, str]], query: str) -> str:
    """Assemble: instruction, then 'Input -> Output' demos, then the query + cue.

    Use a CONSISTENT template, e.g.:
        <instruction>

        Review: <demo input>
        Sentiment: <demo label>
        ...
        Review: <query>
        Sentiment:
    The trailing 'Sentiment:' is the output cue the model should complete.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


def build_zeroshot_prompt(instruction: str, demos, query: str) -> str:
    """A zero-shot prompt ignores demos entirely (provided for comparison)."""
    return f"{instruction}\n\nReview: {query}\nSentiment:"


# ----------------------------- TODO 3 -----------------------------
def run_experiment(model: "Model", prompt_fn: Callable, dataset, demos) -> tuple[list[str], float]:
    """Run prompt_fn over the dataset, query the model, parse labels.

    Returns (predictions, accuracy). `prompt_fn(instruction, demos, query)`
    builds the prompt; use INSTRUCTION below. Parse each model reply with
    `parse_label`.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
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
