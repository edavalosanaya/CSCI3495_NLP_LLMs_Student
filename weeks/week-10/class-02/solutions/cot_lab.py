#!/usr/bin/env python3
"""W10C2 reference solution, chain-of-thought vs. direct prompting."""
from __future__ import annotations
import os
import re
from collections import Counter

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

DATASET: list[tuple[str, int]] = [
    ("A cafe had 23 muffins, sold 17, then baked 12 more. How many now?", 18),
    ("Sam has 5 boxes with 6 pencils each. He gives away 9. How many left?", 21),
    ("A train travels 60 miles in the first hour and 80 in the second. "
     "What is the total distance?", 140),
    ("There are 4 nests with 3 eggs each. 2 eggs hatch. How many unhatched eggs?", 10),
    ("Mia read 15 pages on Monday and twice as many on Tuesday. "
     "How many pages total?", 45),
    ("A bag has 30 marbles. You remove half, then add 4. How many marbles?", 19),
]

# The two problems used for the in-class whiteboard step (see the README).
WARM_UP = DATASET[0]   # cafe muffins -> 23 - 17 + 12 = 18
HARDER = DATASET[3]    # nests/eggs   -> 4 * 3 = 12, then 12 - 2 = 10 unhatched
WHITEBOARD_PROBLEMS = [WARM_UP, HARDER]


def extract_answer(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def majority_vote(answers: list[int]) -> int | None:
    vals = [a for a in answers if a is not None]
    if not vals:
        return None
    counts = Counter(vals)
    top = max(counts.values())
    # tie-break: smallest value among the most common
    return min(v for v, c in counts.items() if c == top)


def evaluate(model: "Model", prompt_fn, dataset) -> float:
    correct = 0
    for question, gold in dataset:
        reply = model.generate(prompt_fn(question))
        if extract_answer(reply) == gold:
            correct += 1
    return correct / len(dataset)


def self_consistency(model: "Model", prompt_fn, dataset, n: int = 5) -> float:
    """Stretch: sample n chains per item (temperature > 0), majority-vote."""
    correct = 0
    for question, gold in dataset:
        answers = [extract_answer(model.generate(prompt_fn(question), temperature=0.7))
                   for _ in range(n)]
        if majority_vote(answers) == gold:
            correct += 1
    return correct / len(dataset)


def direct_prompt(question: str) -> str:
    return f"Answer with only the final number.\n\nQ: {question}\nA:"


def cot_prompt(question: str) -> str:
    return ("Solve the problem. Reason step by step, then end with "
            "'The answer is N.'\n\n"
            f"Q: {question}\nA:")


class Model:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:  # pragma: no cover
        raise NotImplementedError


class StubModel(Model):
    def _correct(self, question: str) -> int | None:
        table = {
            "muffins": 18, "pencils": 21, "train travels": 140,
            "nests": 10, "pages": 45, "marbles": 19,
        }
        for key, val in table.items():
            if key in question.lower():
                return val
        return None

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        qline = next((l for l in prompt.splitlines() if l.strip().startswith("Q:")), "")
        question = qline.split("Q:", 1)[-1].strip()
        correct = self._correct(question)
        wants_reasoning = "step by step" in prompt.lower()
        if correct is None:
            return "0"
        if wants_reasoning:
            return f"Let's work it out carefully. The answer is {correct}."
        return str(correct + 1)


class OllamaModel(Model):
    def __init__(self, name: str = MODEL):
        import ollama
        self.client = ollama.Client()
        self.name = name

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self.client.chat(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "seed": 0, "num_predict": 200},
        )
        return resp["message"]["content"]


def get_model() -> Model:
    try:
        m = OllamaModel()
        m.generate("Q: 1+1?\nA:")
        print(f"[model] using Ollama model '{MODEL}'")
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[model] Ollama unavailable ({type(e).__name__}); using offline stub.")
        return StubModel()


def main() -> int:
    model = get_model()
    print(f"\n{'prompt style':<16}{'accuracy':>10}")
    print("-" * 26)
    for name, fn in {"direct": direct_prompt, "chain-of-thought": cot_prompt}.items():
        acc = evaluate(model, fn, DATASET)
        print(f"{name:<16}{acc:>9.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
