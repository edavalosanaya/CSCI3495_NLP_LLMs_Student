#!/usr/bin/env python3
"""W1C1, Hello, NLP: your first local LLM prompt via Ollama."""
from __future__ import annotations
import os
import sys

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")


def main() -> int:
    try:
        import ollama
    except ImportError:
        print("The 'ollama' package is missing. Run inside the course container.")
        return 1

    client = ollama.Client()  # uses OLLAMA_HOST from the environment

    # --- STEP 4: change this prompt to something you actually want to ask ---
    prompt = "In one short sentence, what is Natural Language Processing?"

    # --- STEP 5: run twice at 0.0, then twice at 1.2, and compare ---
    temperature = 0.7

    try:
        resp = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
    except Exception as e:  # noqa: BLE001
        print(f"Could not reach Ollama / model '{MODEL}': {e}\n")
        print("Fix it with:")
        print("  docker compose -f docker/docker-compose.yml up -d ollama")
        print(f"  docker compose -f docker/docker-compose.yml exec ollama ollama pull {MODEL}")
        return 1

    print(f"Model:       {MODEL}")
    print(f"Temperature: {temperature}")
    print(f"Prompt:      {prompt}")
    print("-" * 60)
    print(resp["message"]["content"].strip())
    print("-" * 60)
    print("Now work through STEPS 4-6 in this file (see README.md).")
    return 0


# --- STEP 6: write your 2-3 sentence reflection here as a comment ---
# Reflection:
#

if __name__ == "__main__":
    raise SystemExit(main())
