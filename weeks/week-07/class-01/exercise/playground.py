#!/usr/bin/env python3
"""W7C1, decoding-strategy playground on a REAL local LLM (via Ollama).

    python playground.py
"""
from __future__ import annotations

import os
import sys

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")
PROMPT = "Write one short, vivid sentence about a city at night."

# (label, options), Ollama options control decoding.
SETTINGS = [
    ("greedy (temp=0)", {"temperature": 0.0}),
    ("low temp (0.3)", {"temperature": 0.3}),
    ("high temp (1.3)", {"temperature": 1.3}),
    ("top_k=5", {"temperature": 1.0, "top_k": 5}),
    ("top_p=0.5", {"temperature": 1.0, "top_p": 0.5}),
    ("top_p=0.95", {"temperature": 1.0, "top_p": 0.95}),
]


def main() -> int:
    try:
        import ollama
    except ImportError:
        print("The 'ollama' package is missing. Run inside the course container.")
        return 0

    client = ollama.Client()  # uses OLLAMA_HOST from the environment

    # Probe availability first so we can skip cleanly.
    try:
        client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_predict": 1},
        )
    except Exception as e:  # noqa: BLE001
        print(f"[skip] Could not reach Ollama / model '{MODEL}': {e}\n")
        print("To run this exercise for real:")
        print("  docker compose -f docker/docker-compose.yml up -d ollama")
        print(f"  docker compose -f docker/docker-compose.yml exec ollama ollama pull {MODEL}")
        print("\n(That's fine, the decoding math is covered by decoding.py + its tests.)")
        return 0

    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT}\n")
    for label, opts in SETTINGS:
        # Seed for reproducibility where supported.
        opts = {"seed": 0, "num_predict": 40, **opts}
        try:
            resp = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT}],
                options=opts,
            )
            text = resp["message"]["content"].strip().replace("\n", " ")
        except Exception as e:  # noqa: BLE001
            text = f"<error: {e}>"
        print(f"--- {label} ---")
        print(text)
        print()

    print("Observe: greedy/low-temp = safe & repetitive; high-temp/top_p=0.95 = diverse.")
    print("Try editing SETTINGS and PROMPT, then re-run!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
