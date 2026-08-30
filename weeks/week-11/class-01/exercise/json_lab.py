#!/usr/bin/env python3
"""W11C1 starter, schema-constrained JSON generation + validation.

Work through the companion-lab section of `README.md`. Each STEP has its own
check (python -m pytest ... -k step1 -q). Or run everything:
    python -m pytest weeks/week-11/class-01/exercise/test_json_lab.py -q

End-to-end (real local model if available, else stub):
    python weeks/week-11/class-01/exercise/json_lab.py
"""
from __future__ import annotations
import json
import os
import re

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

# A tiny schema: field -> spec. type is a python type; optional range/choices.
SCHEMA: dict[str, dict] = {
    "name": {"type": str},
    "price": {"type": (int, float), "min": 0},
    "in_stock": {"type": bool},
    "rating": {"type": int, "min": 1, "max": 5},
}


# ----------------------------- STEP 1 -----------------------------
def extract_json(text: str) -> dict:
    """Return the first JSON object found in `text`, or raise ValueError.

    Models wrap JSON in prose or ```json fences. Find the substring from the
    first '{' to the matching '}' (a simple last-'}' heuristic is fine here),
    then json.loads it. Raise ValueError if there is no valid object.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


# ----------------------------- STEP 2 -----------------------------
def validate(record: dict, schema: dict) -> list[str]:
    """Return a list of error strings (empty == valid).

    For each field in the schema check:
      - present in record
      - correct type (note: in Python bool is a subclass of int, guard for it)
      - within 'min'/'max' if given
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


# ----------------------------- STEP 3 -----------------------------
def generate_valid(model: "Model", schema: dict, max_retries: int = 2) -> dict:
    """Generate -> extract -> validate, re-prompting with errors on failure.

    Return the first valid record. Raise RuntimeError if still invalid after
    max_retries extra attempts. Use build_prompt(schema, error=...).
    """
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    error: str | None = None
    for attempt in range(max_retries + 1):
        reply = model.generate(build_prompt(schema, error))
        try:
            record = extract_json(reply)
        except ValueError as e:
            error = str(e)
            continue
        errors = validate(record, schema)
        if not errors:
            return record
        error = "; ".join(errors)
    raise RuntimeError(f"no valid record after {max_retries + 1} attempts: {error}")


def build_prompt(schema: dict, error: str | None = None) -> str:
    fields = ", ".join(f"{k} ({_spec_str(v)})" for k, v in schema.items())
    base = (
        "Return ONLY a JSON object (no prose, no markdown) with these fields:\n"
        f"  {fields}\n"
        'Example: {"name": "Pen", "price": 1.5, "in_stock": true, "rating": 4}\n'
    )
    if error:
        base += f"\nYour previous answer was invalid: {error}\nReturn corrected JSON only."
    return base


def _spec_str(spec: dict) -> str:
    t = spec["type"]
    name = "number" if t in ((int, float),) else getattr(t, "__name__", str(t))
    rng = ""
    if "min" in spec and "max" in spec:
        rng = f", {spec['min']}-{spec['max']}"
    elif "min" in spec:
        rng = f", >= {spec['min']}"
    return f"{name}{rng}"


# --------------------------- model backends ---------------------------
class Model:
    def generate(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class StubModel(Model):
    """Returns progressively-better canned replies to exercise the retry loop.

    Attempt 1: invalid (price is a string, wrapped in a markdown fence).
    Attempt 2: invalid (rating out of range).
    Attempt 3+: valid JSON.
    """

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return '```json\n{"name": "Mug", "price": "five", "in_stock": true, "rating": 4}\n```'
        if self.calls == 2:
            return 'Here you go: {"name": "Mug", "price": 5.0, "in_stock": true, "rating": 9}'
        return '{"name": "Mug", "price": 5.0, "in_stock": true, "rating": 4}'


class OllamaModel(Model):
    def __init__(self, name: str = MODEL):
        import ollama
        self.client = ollama.Client()
        self.name = name

    def generate(self, prompt: str) -> str:
        resp = self.client.chat(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "seed": 0},
            format="json", # constrained decoding: force syntactically valid JSON
        )
        return resp["message"]["content"]


def get_model() -> Model:
    try:
        m = OllamaModel()
        m.generate("Return {}")
        print(f"[model] using Ollama model '{MODEL}'")
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[model] Ollama unavailable ({type(e).__name__}); using offline stub.")
        return StubModel()


def main() -> int:
    model = get_model()
    try:
        record = generate_valid(model, SCHEMA)
        print("\nValid record obtained:")
        print(json.dumps(record, indent=2))
    except NotImplementedError:
        print("\n(generate_valid is a TODO, implement it to run end-to-end.)")
        return 0
    except RuntimeError as e:
        print(f"\nFailed to obtain a valid record: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
