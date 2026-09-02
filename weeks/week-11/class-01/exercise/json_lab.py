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
def _type_name(t) -> str:
    if t == (int, float):
        return "number"
    return getattr(t, "__name__", str(t))


def extract_json(text: str) -> dict:
    """Dig a JSON object out of whatever the model wrapped it in.

    Args:
        text: raw model output. It may be bare JSON, or JSON inside a ```json
            fence, or JSON with a sentence of prose either side of it.

    Returns:
        The parsed object, as a dict.

    Raises:
        ValueError: if there is no JSON object in there at all, if it does not
            parse, or if it parses to something that is not a dict. A list is
            valid JSON and still wrong here.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   find the first opening brace and the LAST closing brace
    #   if either is missing, or they are the wrong way round, there is no
    #       object to extract
    #   parse the slice between them, turning any parse failure into a
    #       ValueError so callers only have one exception type to handle
    #   reject anything that did not come back as a dict
    #
    #   First-open to last-close is deliberately crude, and enough here.
    #
    raise NotImplementedError


# ----------------------------- STEP 2 -----------------------------
def validate(record: dict, schema: dict) -> list[str]:
    """Check a parsed record against a schema and report everything wrong.

    Args:
        record: the dict that came back from extract_json.
        schema: field name -> spec. Each spec has a "type", and may have "min"
            and "max" bounds.

    Returns:
        A list of human-readable error strings, EMPTY when the record is valid.
        Collect every problem rather than returning at the first: a caller
        fixing a model's output wants the whole list at once.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   for every field the schema asks for:
    #       if the record does not have it at all, record that and move on
    #       if the value is the wrong type, record that and move on
    #       otherwise check it against any min and max the spec gives
    #   hand back everything you found
    #
    #   One trap: in Python a bool IS an int, so True passes an int check
    #   unless you rule it out on purpose. _type_name is given for the
    #   messages.
    #
    #   _type_name is given for the message text.
    #
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
