"""W9C2 starter, build a mini evaluation harness + catch a hallucination + a
position-bias-aware LLM-as-judge.

A benchmark = dataset + metric + protocol. Here you implement the *metric and
protocol* part of a harness in three parts:

PART 1 (factuality scoring): normalize answers, score with exact-match and a
multiple-choice scorer, aggregate accuracy, and flag a hallucination on an
UNANSWERABLE question (the model should abstain, not invent an answer).

PART 2 (LLM-as-judge): when there is no single gold answer (open-ended chat),
a common shortcut is to ask a *strong model to judge*. But judges have biases.
You will build a tiny PAIRWISE judge harness that detects POSITION BIAS by
asking the same comparison twice with the order swapped (A,B then B,A) and only
trusting a verdict that survives the swap.

Work through the lab in `README.md`. Each STEP below has its own check, and
everything is testable WITHOUT any model:
    python -m pytest weeks/week-09/class-02/exercise/test_eval_harness.py -k step1 -q

The optional live runs (`run_model_eval`, `run_judge_demo`) degrade gracefully
if Ollama is down.
"""
from __future__ import annotations

import os
import re
import string

_ARTICLES = {"a", "an", "the"}

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

# A tiny QA slice: each item has a question, the gold answer, and (for the
# hallucination demo) whether it is *answerable*. The last item is a trap: it
# asks about a fictional/future fact a truthful model should refuse.
DATASET = [
    {"q": "What is the capital of France?", "gold": "Paris", "answerable": True},
    {"q": "What is 2 + 2?", "gold": "4", "answerable": True},
    {"q": "What language is primarily spoken in Brazil?", "gold": "Portuguese", "answerable": True},
    {"q": "Who won the Nobel Prize in Physics in the year 2087?", "gold": None, "answerable": False},
]


# ---------------------------------------------------------------------------
# PART 1: factuality scoring
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles (a/an/the), collapse whitespace.

    This is the standard SQuAD-style normalization so 'The Paris.' == 'paris'.
    """
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> bool:
    """True if normalized prediction == normalized gold."""
    # GIVEN (STEP 2): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return normalize_answer(pred) == normalize_answer(gold)


def contains_answer(pred: str, gold: str) -> bool:
    """True if the normalized gold appears as a token-substring of the prediction.

    Models often answer in a sentence ('The capital is Paris.'), so a lenient
    'gold is contained in pred' metric is useful alongside exact match.
    """
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not g:
        return False
    # token-aware substring: match on word boundaries within the normalized pred
    return bool(re.search(r"(?:^|\s)" + re.escape(g) + r"(?:\s|$)", p))


def accuracy(preds: list[str], golds: list[str]) -> float:
    """Fraction of items where contains_answer(pred, gold) is True.

    (Only over items with a non-None gold; raise ValueError if lengths differ.)
    """
    # GIVEN (STEP 4): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if contains_answer(p, g))
    return hits / len(preds)


def is_hallucination(pred: str, item: dict) -> bool:
    """Flag a likely hallucination on an UNANSWERABLE item.

    If item['answerable'] is False, a good model should abstain. We treat the
    answer as a hallucination if the model did NOT abstain -- i.e. the prediction
    contains none of the abstention cues below.
    """
    abstain_cues = (
        "i don't know", "i do not know", "cannot", "can't", "no winner",
        "hasn't happened", "has not happened", "in the future", "not sure",
        "no information", "unable", "fictional", "does not exist", "doesn't exist",
    )
    if item["answerable"]:
        return False
    low = pred.lower()
    # TODO (STEP 5): implement. Check with: pytest -k step5
    raise NotImplementedError


# ---------------------------------------------------------------------------
# PART 2: LLM-as-judge with a position-bias check
# ---------------------------------------------------------------------------
# A "judge" is any function that, given a question and two candidate answers
# labelled "A" and "B", returns one of: "A", "B", or "tie".
#
# `biased_judge` below is a deterministic stand-in for a position-biased model:
# it ALWAYS picks whichever answer is in the first slot. We use it (offline,
# no Ollama needed) to prove that the swap test catches the bias. A real Ollama
# judge can be dropped in later via `ollama_judge`.

def biased_judge(question: str, answer_a: str, answer_b: str) -> str:
    """A deterministic judge that always favors the FIRST slot (position bias)."""
    return "A"


def judge_pairwise(judge, question: str, ans1: str, ans2: str) -> dict:
    """Run the judge BOTH ways and report whether the verdict is consistent.

    Call the judge once with (ans1 as A, ans2 as B) and once with the order
    SWAPPED (ans2 as A, ans1 as B). Translate each raw "A"/"B"/"tie" verdict
    back into which *answer* (1 or 2) it favors, so the two runs are comparable.

    Return a dict:
        {
          "winner_run1": "ans1" | "ans2" | "tie",   # who won when ans1 was shown first
          "winner_run2": "ans1" | "ans2" | "tie",   # who won when ans2 was shown first
          "consistent": bool,                        # same winner both ways?
        }

    A robust harness only trusts a verdict when consistent is True.
    """
    # TODO (STEP 6): implement. Check with: pytest -k step6
    #   1) raw1 = judge(question, ans1, ans2); map "A"->"ans1", "B"->"ans2", "tie"->"tie".
    #   2) raw2 = judge(question, ans2, ans1); now "A"->"ans2", "B"->"ans1", "tie"->"tie".
    #   3) consistent = (winner_run1 == winner_run2).
    raise NotImplementedError


def position_bias_rate(judge, pairs: list[tuple[str, str, str]]) -> float:
    """Fraction of pairs whose verdict is INCONSISTENT under the swap.

    `pairs` is a list of (question, ans1, ans2). A perfectly fair judge scores
    0.0; a fully position-biased judge (always picks the first slot) scores 1.0
    on pairs where it would otherwise have to choose.
    """
    # TODO (STEP 7): implement. Check with: pytest -k step7
    #   run judge_pairwise on each pair; return the fraction with
    #   consistent == False. (Empty list -> 0.0)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Optional live runs against Ollama (degrade gracefully if unavailable)
# ---------------------------------------------------------------------------
def run_model_eval(model: str = MODEL) -> int:
    """Optional live run against Ollama; degrades gracefully if unavailable."""
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
    """Build a real LLM judge backed by Ollama (used only by run_judge_demo)."""
    import ollama  # imported lazily so offline tests never need it
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
    """Offline-first demo of the swap test with the deterministic biased judge,
    plus an optional live judge if Ollama is reachable."""
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
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        run_model_eval()
        print("\n" + "=" * 60 + "\n")
        raise SystemExit(run_judge_demo())
    except NotImplementedError:
        print("eval_harness.py is not finished yet: fill in the next TODO in this file, then re-run.")
