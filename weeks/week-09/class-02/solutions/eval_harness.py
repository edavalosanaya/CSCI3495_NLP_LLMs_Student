"""W9C2 reference solution, mini evaluation harness + hallucination flag +
position-bias-aware LLM-as-judge."""
from __future__ import annotations

import os
import re
import string

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

DATASET = [
    {"q": "What is the capital of France?", "gold": "Paris", "answerable": True},
    {"q": "What is 2 + 2?", "gold": "4", "answerable": True},
    {"q": "What language is primarily spoken in Brazil?", "gold": "Portuguese", "answerable": True},
    {"q": "Who won the Nobel Prize in Physics in the year 2087?", "gold": None, "answerable": False},
]

_ARTICLES = {"a", "an", "the"}


# ---------------------------------------------------------------------------
# PART 1: factuality scoring
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_answer(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not g:
        return False
    # token-aware substring: match on word boundaries within the normalized pred
    return bool(re.search(r"(?:^|\s)" + re.escape(g) + r"(?:\s|$)", p))


def accuracy(preds: list[str], golds: list[str]) -> float:
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if contains_answer(p, g))
    return hits / len(preds)


def is_hallucination(pred: str, item: dict) -> bool:
    abstain_cues = (
        "i don't know", "i do not know", "cannot", "can't", "no winner",
        "hasn't happened", "has not happened", "in the future", "not sure",
        "no information", "unable", "fictional", "does not exist", "doesn't exist",
    )
    if item["answerable"]:
        return False
    low = pred.lower()
    return not any(cue in low for cue in abstain_cues)


# ---------------------------------------------------------------------------
# PART 2: LLM-as-judge with a position-bias check
# ---------------------------------------------------------------------------
def biased_judge(question: str, answer_a: str, answer_b: str) -> str:
    """A deterministic judge that always favors the FIRST slot (position bias)."""
    return "A"


def _winner_of(raw: str, first: str, second: str) -> str:
    """Map a raw 'A'/'B'/'tie' verdict back to which *answer* it favors."""
    if raw == "A":
        return first
    if raw == "B":
        return second
    return "tie"


def judge_pairwise(judge, question: str, ans1: str, ans2: str) -> dict:
    raw1 = judge(question, ans1, ans2)          # ans1 in slot A, ans2 in slot B
    raw2 = judge(question, ans2, ans1)          # swapped: ans2 in slot A
    winner_run1 = _winner_of(raw1, "ans1", "ans2")
    winner_run2 = _winner_of(raw2, "ans2", "ans1")
    return {
        "winner_run1": winner_run1,
        "winner_run2": winner_run2,
        "consistent": winner_run1 == winner_run2,
    }


def position_bias_rate(judge, pairs: list[tuple[str, str, str]]) -> float:
    if not pairs:
        return 0.0
    inconsistent = sum(
        0 if judge_pairwise(judge, q, a1, a2)["consistent"] else 1
        for q, a1, a2 in pairs
    )
    return inconsistent / len(pairs)


# ---------------------------------------------------------------------------
# Optional live runs against Ollama (degrade gracefully if unavailable)
# ---------------------------------------------------------------------------
def run_model_eval(model: str = MODEL) -> int:
    try:
        import ollama
    except ImportError:
        print("The 'ollama' package is missing. Run inside the course container.")
        return 1
    client = ollama.Client()
    preds, golds = [], []
    print(f"Evaluating {model} on {len(DATASET)} items\n" + "-" * 60)
    for item in DATASET:
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": item["q"]}],
                options={"temperature": 0.0},
            )
            pred = resp["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            print(f"Could not reach Ollama / model '{model}': {e}")
            print("Skipping live eval. The scoring functions are still tested offline.")
            print("Start it with:")
            print("  docker compose -f docker/docker-compose.yml up -d ollama")
            print(f"  docker compose -f docker/docker-compose.yml exec ollama ollama pull {model}")
            return 1
        flag = " <-- HALLUCINATION?" if is_hallucination(pred, item) else ""
        print(f"Q: {item['q']}\nA: {pred[:120]}{flag}\n")
        if item["gold"] is not None:
            preds.append(pred)
            golds.append(item["gold"])
    print("-" * 60)
    print(f"Accuracy on answerable items: {accuracy(preds, golds):.2%}")
    return 0


def ollama_judge(model: str = MODEL):
    import ollama
    client = ollama.Client()

    def _judge(question: str, answer_a: str, answer_b: str) -> str:
        prompt = (
            "You are grading two answers to a question. Reply with EXACTLY one "
            "token: A, B, or tie.\n\n"
            f"Question: {question}\n\n[A] {answer_a}\n\n[B] {answer_b}\n\nVerdict:"
        )
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        out = resp["message"]["content"].strip().lower()
        if out.startswith("a"):
            return "A"
        if out.startswith("b"):
            return "B"
        return "tie"

    return _judge


def run_judge_demo(model: str = MODEL) -> int:
    pairs = [
        ("Explain why the sky is blue.",
         "Sunlight scatters off air molecules; short blue wavelengths scatter most.",
         "The sky is blue because of the ocean reflecting upward, more or less."),
        ("What is a good study tip?",
         "Space your practice over days and self-test; it beats cramming.",
         "Just read the textbook many times the night before the exam."),
    ]
    print("== Deterministic biased judge (always picks the first slot) ==")
    for q, a1, a2 in pairs:
        r = judge_pairwise(biased_judge, q, a1, a2)
        print(f"Q: {q}\n  run1 winner: {r['winner_run1']}  run2 winner: {r['winner_run2']}"
              f"  consistent: {r['consistent']}")
    rate = position_bias_rate(biased_judge, pairs)
    print(f"Position-bias (inconsistency) rate: {rate:.0%}\n")

    try:
        judge = ollama_judge(model)
        print(f"== Live judge: {model} (with swap check) ==")
        for q, a1, a2 in pairs:
            r = judge_pairwise(judge, q, a1, a2)
            print(f"Q: {q}\n  run1 winner: {r['winner_run1']}  run2 winner: {r['winner_run2']}"
                  f"  consistent: {r['consistent']}")
        print(f"Position-bias (inconsistency) rate: {position_bias_rate(judge, pairs):.0%}")
    except Exception as e:  # noqa: BLE001
        print(f"(Skipping live judge: {e})")
        print("Start Ollama to try a real judge:")
        print("  docker compose -f docker/docker-compose.yml up -d ollama")
        print(f"  docker compose -f docker/docker-compose.yml exec ollama ollama pull {model}")
    return 0


if __name__ == "__main__":
    run_model_eval()
    print("\n" + "=" * 60 + "\n")
    raise SystemExit(run_judge_demo())
