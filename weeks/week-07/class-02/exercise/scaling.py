"""W7C2 starter, measure scaling behavior across model sizes.

Idea: run the SAME small task suite on two or three Ollama models of different
sizes (e.g. qwen2.5:0.5b vs. llama3.2:1b) and see whether the bigger model does
better. That's scaling, observed on a laptop.

You implement the pure-Python *scoring* core (fully tested, no model needed).
`measure.py` then drives real models via Ollama and skips cleanly if absent.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-07/class-02/exercise/test_scaling.py -k step1 -q
"""
from __future__ import annotations
import string

# A tiny task suite: simple factual / arithmetic questions with a target string
# that should appear in a correct answer. (Lenient substring grading.)
TASKS = [
    {"q": "What is 2 + 2? Answer with just the number.", "answer": "4"},
    {"q": "What is the capital of France? One word.", "answer": "paris"},
    {"q": "How many days are in a week? Just the number.", "answer": "7"},
    {"q": "What color do you get mixing blue and yellow? One word.", "answer": "green"},
    {"q": "What is 10 minus 3? Just the number.", "answer": "7"},
]


def normalize(text: str) -> str:
    """Lowercase and strip surrounding whitespace/punctuation for lenient matching."""
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return text.strip().strip(string.punctuation + " ").lower()


def is_correct(model_output: str, target: str) -> bool:
    """True if the normalized target appears as a substring of the normalized output."""
    # GIVEN (STEP 2): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return normalize(target) in normalize(model_output)


def accuracy(outputs: list[str], targets: list[str]) -> float:
    """Fraction of items where is_correct() is True. Returns a float in [0, 1]."""
    # TODO (STEP 3): implement. Check with: pytest -k step3
    raise NotImplementedError


def scaling_trend(results: dict[str, float]) -> bool:
    """Given {model_name: accuracy} ordered from SMALLEST to LARGEST model,
    return True if accuracy is (weakly) non-decreasing with size, i.e. the
    bigger models did at least as well. This is the 'scaling helps' check.
    """
    # TODO (STEP 4): implement. Check with: pytest -k step4
    raise NotImplementedError


def _demo() -> None:
    # Simulated outputs (what a tiny vs. a bigger model might say).
    targets = [t["answer"] for t in TASKS]
    small = ["4", "Lyon", "7", "green", "six"]          # 3/5
    large = ["4", "Paris.", "7 days", "Green", "7"]     # 5/5
    acc_small = accuracy(small, targets)
    acc_large = accuracy(large, targets)
    print(f"small-model accuracy: {acc_small:.2f}")
    print(f"large-model accuracy: {acc_large:.2f}")
    trend = scaling_trend({"small": acc_small, "large": acc_large})
    print(f"accuracy non-decreasing with size? {trend}")


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        _demo()
    except NotImplementedError:
        print("scaling.py is not finished yet: fill in the next TODO in this file, then re-run.")
