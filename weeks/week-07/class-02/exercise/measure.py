#!/usr/bin/env python3
"""W7C2, measure scaling behavior across real model sizes (via Ollama).

    python measure.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"scaling_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_impl(mod) -> bool:
    try:
        mod.normalize("x")
        return True
    except Exception:  # noqa: BLE001
        return False


# Use the student's scoring core if implemented; else fall back to the reference
# solution so this script always works as a demo.
sc = _load(_HERE / "scaling.py")
if not _has_impl(sc):
    sc = _load(_HERE.parent / "solutions" / "scaling.py")


# Ordered SMALLEST -> LARGEST so a non-decreasing accuracy = "scaling helps".
MODELS = [
    m.strip()
    for m in os.environ.get("SCALING_MODELS", "qwen2.5:0.5b,llama3.2:1b").split(",")
    if m.strip()
]


def ask(client, model: str, question: str) -> str:
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.0, "seed": 0, "num_predict": 20},
    )
    return resp["message"]["content"].strip()


def main() -> int:
    try:
        import ollama
    except ImportError:
        print("The 'ollama' package is missing. Run inside the course container.")
        return 0

    client = ollama.Client()
    targets = [t["answer"] for t in sc.TASKS]
    results: dict[str, float] = {}

    for model in MODELS:
        try:
            outputs = [ask(client, model, t["q"]) for t in sc.TASKS]
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {model}: not available ({e})")
            continue
        acc = sc.accuracy(outputs, targets)
        results[model] = acc
        print(f"{model:<18} accuracy = {acc:.2f}")
        for t, o in zip(sc.TASKS, outputs):
            mark = "ok " if sc.is_correct(o, t["answer"]) else "  x"
            print(f"   [{mark}] {t['q'][:42]:<42} -> {o[:30]!r}")

    if not results:
        print("\nNo models reachable. To run this for real:")
        print("  docker compose -f docker/docker-compose.yml up -d ollama")
        for m in MODELS:
            print(f"  docker compose -f docker/docker-compose.yml exec ollama ollama pull {m}")
        print("\n(That's fine, the scoring core is covered by scaling.py + its tests.)")
        return 0

    if len(results) >= 2:
        trend = sc.scaling_trend(results)
        print(f"\nAccuracy non-decreasing from smallest to largest model? {trend}")
        print("If True, you just watched scaling help on your own laptop.")
    return 0


if __name__ == "__main__":
    # A student running this before finishing scaling.py should see a
    # sentence, not a traceback: an unwritten step is a normal state.
    try:
        raise SystemExit(main())
    except NotImplementedError:
        print("scaling.py is not finished yet: fill in the next TODO there, then re-run.")
