#!/usr/bin/env python3
"""W11C1 reference solution, schema-constrained JSON generation + validation."""
from __future__ import annotations
import json
import os
import re

MODEL = os.environ.get("COURSE_MODEL", "qwen2.5:0.5b")

SCHEMA: dict[str, dict] = {
    "name": {"type": str},
    "price": {"type": (int, float), "min": 0},
    "in_stock": {"type": bool},
    "rating": {"type": int, "min": 1, "max": 5},
}


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in text")
    snippet = text[start:end + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def validate(record: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    for field, spec in schema.items():
        if field not in record:
            errors.append(f"{field}: missing required field")
            continue
        value = record[field]
        expected = spec["type"]
        # bool is a subclass of int, reject bools where a number/int is expected.
        if expected is not bool and isinstance(value, bool):
            errors.append(f"{field}: expected {_type_name(expected)}, got bool")
            continue
        if not isinstance(value, expected):
            errors.append(f"{field}: expected {_type_name(expected)}, got {type(value).__name__}")
            continue
        if "min" in spec and value < spec["min"]:
            errors.append(f"{field}: {value} below minimum {spec['min']}")
        if "max" in spec and value > spec["max"]:
            errors.append(f"{field}: {value} above maximum {spec['max']}")
    return errors


def _type_name(t) -> str:
    if t == (int, float):
        return "number"
    return getattr(t, "__name__", str(t))


def generate_valid(model: "Model", schema: dict, max_retries: int = 2) -> dict:
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
    name = _type_name(spec["type"])
    if "min" in spec and "max" in spec:
        return f"{name}, {spec['min']}-{spec['max']}"
    if "min" in spec:
        return f"{name}, >= {spec['min']}"
    return name


class Model:
    def generate(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class StubModel(Model):
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
            format="json",
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
    except RuntimeError as e:
        print(f"\nFailed to obtain a valid record: {e}")
        return 1
    print("\nValid record obtained:")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
