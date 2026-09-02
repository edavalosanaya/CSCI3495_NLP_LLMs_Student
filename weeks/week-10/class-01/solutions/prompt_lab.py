#!/usr/bin/env python3
"""W10C1 reference solution, prompt-experiment harness.

Measured on qwen2.5:0.5b (temperature 0, seed 0, num_predict 8), 8-item set:

    variant                   words  accuracy
    zero-shot (baseline)         17       88%
    few-shot (baseline)          30       50%   <- MORE demos, WORSE score
    few-shot + constraint        33      100%
    zero-shot + constraint       20      100%
    golf: 'One word only.'       11      100%
    golf too far                  8       12%

The lesson: the baseline few-shot prompt does not fail at sentiment, it fails
at FORMAT. Asked to "classify", the model starts a sentence ("The sentiment of
this review is ...") that num_predict=8 truncates, and in the few-shot layout it
latches onto "Positive" for every item. One clause pinning the output shape
("Answer with one word: positive or negative.") takes it to 100%, and the same
constraint golfs down to two words before accuracy falls off a cliff.
"""
from __future__ import annotations
import os
import re
from typing import Callable

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

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


def accuracy(preds: list[str], golds: list[str]) -> float:
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(golds)


def build_fewshot_prompt(instruction: str, demos: list[tuple[str, str]], query: str) -> str:
    lines = [instruction, ""]
    for text, label in demos:
        lines.append(f"Review: {text}")
        lines.append(f"Sentiment: {label}")
        lines.append("")
    lines.append(f"Review: {query}")
    lines.append("Sentiment:")
    return "\n".join(lines)


def build_zeroshot_prompt(instruction: str, demos, query: str) -> str:
    return f"{instruction}\n\nReview: {query}\nSentiment:"


def run_experiment(model: "Model", prompt_fn: Callable, dataset, demos,
                   instruction: str | None = None) -> tuple[list[str], float]:
    if instruction is None:
        instruction = INSTRUCTION

    preds = []
    golds = []
    for text, gold in dataset:
        prompt = prompt_fn(instruction, demos, text)
        reply = model.generate(prompt)
        preds.append(parse_label(reply))
        golds.append(gold)

    return preds, accuracy(preds, golds)


INSTRUCTION = "Classify the movie review's sentiment as positive or negative."

# Three prompts that BEAT the baseline on qwen2.5:0.5b. All three win the same
# way: they constrain the OUTPUT FORMAT. The baseline's failure is not a
# reasoning failure, it is a formatting failure (see the module docstring).
CONSTRAINED = ("Classify the movie review's sentiment. "
               "Answer with one word: positive or negative.")
GOLFED = "One word only."


def build_bare_prompt(instruction, demos, query: str) -> str:
    """Golf taken one step too far: the cues alone, no instruction at all."""
    return f"Review: {query}\nSentiment:"


def parse_label(text: str) -> str:
    t = text.lower()
    pos = re.search(r"positive|good|favorable", t)
    neg = re.search(r"negative|bad|unfavorable", t)
    if pos and (not neg or pos.start() < neg.start()):
        return "positive"
    if neg:
        return "negative"
    return "unknown"


class Model:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class StubModel(Model):
    POS = {"loved", "superb", "best", "charming", "funny", "heartfelt", "delightful", "moving"}
    NEG = {"waste", "boring", "predictable", "money", "back", "sense", "dull", "forgettable"}

    def generate(self, prompt: str) -> str:
        lines = [ln for ln in prompt.splitlines() if ln.strip().lower().startswith("review:")]
        query = lines[-1] if lines else prompt
        words = set(re.findall(r"[a-z']+", query.lower()))
        score = len(words & self.POS) - len(words & self.NEG)
        return "positive" if score >= 0 else "negative"


class OllamaModel(Model):
    def __init__(self, name: str = MODEL):
        import ollama
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
    try:
        m = OllamaModel()
        m.generate("Review: test\nSentiment:")
        print(f"[model] using Ollama model '{MODEL}'")
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[model] Ollama unavailable ({type(e).__name__}); using offline stub.")
        print("        Start it with: docker compose ... up -d ollama && ollama pull " + MODEL)
        return StubModel()


def main() -> int:
    model = get_model()

    # (label, instruction, prompt builder). The first two are the baselines
    # that disappoint; the last three are the fixes worth showing the class.
    experiments = [
        ("zero-shot (baseline)", INSTRUCTION, build_zeroshot_prompt),
        ("few-shot (baseline)", INSTRUCTION, build_fewshot_prompt),
        ("few-shot + constraint", CONSTRAINED, build_fewshot_prompt),
        ("zero-shot + constraint", CONSTRAINED, build_zeroshot_prompt),
        ("golf: 'One word only.'", GOLFED, build_zeroshot_prompt),
        ("golf too far", "", build_bare_prompt),
    ]
    print(f"\n{'variant':<24}{'words':>7}{'accuracy':>10}")
    print("-" * 41)
    for name, instruction, fn in experiments:
        n_words = len(fn(instruction, DEMOS, DATASET[0][0]).split())
        _, acc = run_experiment(model, fn, DATASET, DEMOS, instruction)
        print(f"{name:<24}{n_words:>7}{acc:>9.0%}")
    print("\nPrompt golf: fewest words that still hold the target accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
