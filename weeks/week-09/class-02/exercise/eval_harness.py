"""W9C2 starter, an evaluation harness with a position-bias-aware judge."""
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
def _winner_of(raw: str, first: str, second: str) -> str:
    """Map a raw 'A'/'B'/'tie' verdict back to which *answer* it favors."""
    if raw == "A":
        return first
    if raw == "B":
        return second
    return "tie"


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles (a/an/the), collapse whitespace.

    This is the standard SQuAD-style normalization so 'The Paris.' == 'paris'.
    """
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> bool:
    """True if normalized prediction == normalized gold."""
    return normalize_answer(pred) == normalize_answer(gold)


def contains_answer(pred: str, gold: str) -> bool:
    """True if the normalized gold appears as a token-substring of the prediction.

    Models often answer in a sentence ('The capital is Paris.'), so a lenient
    'gold is contained in pred' metric is useful alongside exact match.
    """
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
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if contains_answer(p, g))
    return hits / len(preds)


def is_hallucination(pred: str, item: dict) -> bool:
    """Flag a likely hallucination: a confident answer to an unanswerable question.

    Args:
        pred: the model's raw answer text, in whatever case it produced.
        item: one eval item. Only item["answerable"] matters here: False means
            the question has no correct answer and the model should have said so.

    Returns:
        True only when the model answered an unanswerable question without
        abstaining. An answerable item is never flagged, however wrong the
        prediction is: this measures fabrication, not accuracy.
    """
    abstain_cues = (
        "i don't know", "i do not know", "cannot", "can't", "no winner",
        "hasn't happened", "has not happened", "in the future", "not sure",
        "no information", "unable", "fictional", "does not exist", "doesn't exist",
    )
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   a question that HAS an answer can never be a hallucination here, so
    #       those items are never flagged
    #   otherwise the model should have refused to answer, so treat it as a
    #       hallucination unless its text carries one of the abstention cues
    #       listed above
    #   compare case-insensitively: models capitalize unpredictably
    #
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

    Args:
        judge: callable (question, slot_A, slot_B) -> a raw verdict string.
            It is asked twice, so it must be safe to call more than once.
        question: the question both answers are responding to.
        ans1: the first answer.
        ans2: the second answer.

    Returns:
        {"winner_run1", "winner_run2", "consistent"}. The two winners are
        "ans1", "ans2" or "tie", named after the ANSWER not the slot, so they
        are directly comparable across the swap. consistent is True when both
        runs name the same winner, and only then is the verdict worth trusting.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   ask the judge once with ans1 in the first slot
    #   ask it again with ans2 in the first slot
    #   translate each raw verdict into which ANSWER won, using the given
    #       _winner_of helper. Mind the argument order on the second call:
    #       the answer sitting in slot A is different that time
    #   report both winners and whether they agree
    #
    #   Translating slot -> answer is the whole trick. Comparing raw verdicts
    #   would say a position-biased judge was perfectly consistent.
    #
    raise NotImplementedError


def position_bias_rate(judge, pairs: list[tuple[str, str, str]]) -> float:
    """How often this judge changes its mind when the answers are swapped.

    Args:
        judge: the judge under test.
        pairs: (question, ans1, ans2) triples to test it on.

    Returns:
        The fraction of pairs that came back inconsistent, in [0, 1]. A fair
        judge scores 0.0; one that always picks whatever is in the first slot
        scores 1.0. An empty list scores 0.0, not an error.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   guard the empty case
    #   for each pair, run the two-way comparison from step 2 and count the
    #       ones it reports as inconsistent
    #   divide by how many pairs there were
    #
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
