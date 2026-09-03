#!/usr/bin/env python3
"""W10C2 starter, chain-of-thought vs. direct prompting harness."""
from __future__ import annotations
import os
import re
from collections import Counter

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

# Multi-step word problems: (question, integer answer).
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
# WARM_UP is a straight 2-step chain; HARDER hides a trap (the question asks for
# the UNHATCHED eggs, so you must subtract from the product, not report either
# number in the problem). Draw the decomposition tree for HARDER.
WARM_UP = DATASET[0]   # cafe muffins -> 23 - 17 + 12 = 18
HARDER = DATASET[3]    # nests/eggs   -> 4 * 3 = 12, then 12 - 2 = 10 unhatched
WHITEBOARD_PROBLEMS = [WARM_UP, HARDER]


# ----------------------------- TODO 1 -----------------------------
def extract_answer(text: str) -> int | None:
    """Return the LAST integer appearing in `text`, or None if there is none.

    CoT replies put the final number at the end (e.g. 'the answer is 18').
    Hint: re.findall(r'-?\\d+', text) then take the last match.
    """
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


# ----------------------------- TODO 2 -----------------------------
def majority_vote(answers: list[int]) -> int | None:
    """Pick the answer that the most reasoning chains agreed on.

    Args:
        answers: one extracted answer per sampled chain. A chain that produced
            no parsable number contributes None, and those are ignored rather
            than counted as a vote.

    Returns:
        The winning value, or None if every chain failed. Ties go to the
        SMALLEST value, so the same input always gives the same answer.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The vote is the formula in README section 3.
    #
    #   throw away the chains that produced nothing; if none are left, there
    #       is no answer to give
    #   count how many chains voted for each value
    #   find the highest count, and among the values that reached it, return
    #       the smallest
    #
    #   Sorting the tie deliberately is what stops two identical runs from
    #   disagreeing.
    #
    raise NotImplementedError


# ----------------------------- TODO 3 -----------------------------
def evaluate(model: "Model", prompt_fn, dataset) -> float:
    """Score one prompting strategy on the whole dataset.

    Args:
        model: anything with .generate(prompt) -> str.
        prompt_fn: called as prompt_fn(question). Swapping this between
            direct_prompt and cot_prompt is the experiment.
        dataset: (question, gold integer answer) pairs.

    Returns:
        Exact-match accuracy in [0, 1]. A chain that produced no parsable
        number simply counts as wrong.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   for each question and its gold answer:
    #       build the prompt and ask the model
    #       pull the integer out of the reply with the given extract_answer,
    #           and count it right only if it matches the gold exactly
    #   divide by how many questions there were
    #
    raise NotImplementedError


def direct_prompt(question: str) -> str:
    """Ask for the number only, no reasoning."""
    return (f"Answer with only the final number.\n\n"
            f"Q: {question}\nA:")


def cot_prompt(question: str) -> str:
    """Ask the model to reason step by step, then state the answer."""
    return (f"Solve the problem. Reason step by step, then end with "
            f"'The answer is N.'\n\n"
            f"Q: {question}\nA:")


# --------------------------- model backends ---------------------------
class Model:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:  # pragma: no cover
        raise NotImplementedError


class StubModel(Model):
    """Deterministic stand-in that simulates the CoT effect.

    It computes the correct answer from the question, but only *reveals* it when
    the prompt asks for step-by-step reasoning; otherwise it returns a plausible
    wrong guess. This lets the offline pipeline demonstrate CoT > direct.
    """

    def _correct(self, question: str) -> int | None:
        # Hard-coded answers keyed by a distinctive question substring.
        table = {
            "muffins": 18, "pencils": 21, "train travels": 140,
            "nests": 10, "pages": 45, "marbles": 19,
        }
        for key, val in table.items():
            if key in question.lower():
                return val
        return None

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # The question is on the 'Q:' line.
        qline = next((l for l in prompt.splitlines() if l.strip().startswith("Q:")), "")
        question = qline.split("Q:", 1)[-1].strip()
        correct = self._correct(question)
        wants_reasoning = "step by step" in prompt.lower()
        if correct is None:
            return "0"
        if wants_reasoning:
            return f"Let's work it out carefully. The answer is {correct}."
        # Direct mode: a plausible-but-wrong off-by-something guess.
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
        try:
            acc = evaluate(model, fn, DATASET)
        except NotImplementedError:
            print(f"{name:<16}{'(TODO)':>10}")
            continue
        print(f"{name:<16}{acc:>9.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
