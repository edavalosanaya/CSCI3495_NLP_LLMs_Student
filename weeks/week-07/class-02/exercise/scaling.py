"""W7C2 starter, measure scaling behavior across model sizes."""
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
    """GIVEN. Lowercase, and strip surrounding whitespace and punctuation."""
    return text.strip().strip(string.punctuation + " ").lower()


def is_correct(model_output: str, target: str) -> bool:
    """GIVEN. True if the normalized target appears anywhere in the output."""
    return normalize(target) in normalize(model_output)


def accuracy(outputs: list[str], targets: list[str]) -> float:
    """What fraction of the model's answers were right.

    Args:
        outputs: what the model said, one string per question.
        targets: the expected answer for each, same length and same order.

    Returns:
        A float in [0, 1]. An empty run scores 0.0 rather than dividing by
        zero: no questions asked is not the same as every question right.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   guard the empty case first
    #   pair each output with its target and count how many is_correct accepts
    #   divide that count by how many questions there were
    #
    raise NotImplementedError


def scaling_trend(results: dict[str, float]) -> bool:
    """Did accuracy hold up or improve as the models got bigger?

    Args:
        results: model name -> accuracy, ALREADY ordered smallest model first.
            The order is the caller's promise; this function does not sort.

    Returns:
        True if accuracy never drops as you walk that order. Equal accuracies
        count as holding up, so a flat run is True: the claim being tested is
        "bigger did not do worse", not "bigger did better".
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   take the accuracies in the order they were given
    #   the answer is yes exactly when no accuracy is smaller than the one
    #       before it
    #
    #   One line, with a single all(...) over neighbouring pairs.
    #
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
