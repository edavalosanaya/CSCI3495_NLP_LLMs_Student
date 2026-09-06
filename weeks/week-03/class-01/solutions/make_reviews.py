#!/usr/bin/env python3
"""Build the W3C1 corpus: 240 short movie reviews with metadata, seeded.

Roughly a fifth of them praise the film and reverse in the LAST sentence, which
is where the true label comes from. A bag-of-words classifier cannot see that
reversal, so those are the ones it gets wrong. That is deliberate: it holds
accuracy near 0.85 instead of a meaningless 1.00.
"""
import csv
import random
from pathlib import Path

SEED = 3495
N_ROWS = 240
MIXED_RATE = 0.18

POS_OPEN = [
    "I loved every minute of this one",
    "This one completely won me over",
    "Easily the best thing I have seen this year",
    "A quiet, confident little film",
    "I went in expecting nothing and came out delighted",
    "Two hours that flew past",
    "This is the kind of film that stays with you",
    "A warm, generous film",
    "I have already recommended it to three people",
    "Worth every cent of the ticket",
    "A genuine surprise from start to finish",
    "I would happily watch this again tomorrow",
]
POS_BODY = [
    "the acting is superb and the script never wastes a line",
    "the lead performance is quietly brilliant",
    "the pacing is perfect and nothing feels padded",
    "the writing is sharp and the jokes land",
    "the score is gorgeous and the photography even better",
    "every character feels like a real person",
    "the ending earns its emotion instead of demanding it",
    "the direction is confident and the editing is tight",
    "the dialogue is clever without showing off",
    "the whole cast is excellent, not just the leads",
    "it is funny, it is moving, and it knows when to stop",
    "the story builds beautifully to a satisfying finish",
]
POS_VERDICT = [
    "Highly recommended",
    "A real treat",
    "Go and see it",
    "One of my favourites now",
    "I will be watching it again",
    "Absolutely worth your evening",
]

NEG_OPEN = [
    "I wanted my two hours back",
    "What a slog this was",
    "I have rarely been so bored in a cinema",
    "A complete waste of a good cast",
    "This one fell apart early and never recovered",
    "I checked my watch four times",
    "An expensive, empty film",
    "I could not wait for it to end",
    "A tedious, self-important mess",
    "I left before the credits finished",
    "Nothing here works",
    "A dull, forgettable evening",
]
NEG_BODY = [
    "the plot goes nowhere and the dialogue is wooden",
    "the acting is flat and the script is worse",
    "the pacing drags and every scene runs too long",
    "the jokes fall flat and the timing is off",
    "the characters are thin and impossible to care about",
    "the ending is unearned and frankly insulting",
    "the effects are cheap and the editing is a mess",
    "it borrows from better films and improves on none of them",
    "the writing is lazy and the twists are obvious",
    "the lead is miscast and the supporting cast is wasted",
    "it is loud, long and completely hollow",
    "the story collapses the moment you think about it",
]
NEG_VERDICT = [
    "Avoid it",
    "Skip this one",
    "Not worth the ticket",
    "I cannot recommend it",
    "Save your money",
    "A hard pass from me",
]

# The reversal that carries the true label in a mixed review.
TURNS = ["And yet", "Even so", "All the same", "Somehow", "In the end"]

GENRES = ["comedy", "drama", "horror", "scifi"]


def sentence(pool: list[str], rng: random.Random) -> str:
    return rng.choice(pool).capitalize() + "."


def lower_first(text: str) -> str:
    """Lowercase the opening letter, unless it is the pronoun "I"."""
    if text.startswith("I "):
        return text
    return text[0].lower() + text[1:]


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for i in range(N_ROWS):
        label = "pos" if i % 2 == 0 else "neg"
        mixed = rng.random() < MIXED_RATE

        if mixed:
            # The body reads as the OPPOSITE of the label; the last line flips it.
            if label == "pos":
                surface_open, surface_body, verdict = NEG_OPEN, NEG_BODY, POS_VERDICT
            else:
                surface_open, surface_body, verdict = POS_OPEN, POS_BODY, NEG_VERDICT
            parts = [
                rng.choice(surface_open) + ".",
                sentence(surface_body, rng),
                f"{rng.choice(TURNS)}, {lower_first(rng.choice(verdict))}.",
            ]
        else:
            if label == "pos":
                opener, body, verdict = POS_OPEN, POS_BODY, POS_VERDICT
            else:
                opener, body, verdict = NEG_OPEN, NEG_BODY, NEG_VERDICT
            parts = [rng.choice(opener) + ".", sentence(body, rng)]
            if rng.random() < 0.6:
                parts.append(rng.choice(verdict) + ".")

        base = 8 if label == "pos" else 3
        rows.append({
            "review_id": f"r{i + 1:03d}",
            "text": " ".join(parts),
            "label": label,
            "rating": max(1, min(10, base + rng.randint(-2, 2))),
            "genre": rng.choice(GENRES),
            "year": rng.randint(2015, 2025),
        })
    rng.shuffle(rows)
    return rows


def main() -> None:
    rows = build()
    out = Path(__file__).with_name("reviews.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} with {len(rows)} rows")
    for r in rows[:3]:
        print(" ", r["label"], r["rating"], r["genre"], "|", r["text"])


if __name__ == "__main__":
    main()
