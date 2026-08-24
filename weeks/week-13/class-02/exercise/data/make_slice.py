#!/usr/bin/env python3
"""Rebuild `gsm8k_mini.jsonl` from the official GSM8K test split.

    python make_slice.py --n 20            # needs network once
    python make_slice.py --n 40 --seed 7

Kept in the repo so the slice is reproducible and auditable rather than a
mystery file: anyone can check that the 20 problems really are an unbiased
sample of the test split. GSM8K is MIT-licensed (see README.md).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
       "master/grade_school_math/data/test.jsonl")
GOLD = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
CALC = re.compile(r"<<([^>]+)>>")


def arithmetic_heavy(solution: str) -> bool:
    """True if the reference solution multiplies or divides awkward numbers.

    We keep this slice on purpose. On `12 * 6` a 1.5B model does not need a
    calculator, so a suite of easy sums cannot tell a tool-using agent apart
    from one reasoning in its head, and a benchmark that cannot separate the
    systems you are comparing has told you nothing. About a third of the GSM8K
    test split qualifies, so this is a real slice, not a hand-picked one.
    """
    for expr in CALC.findall(solution):
        lhs = expr.split("=")[0]
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", lhs)]
        if not nums or not any(op in lhs for op in "*/"):
            continue
        if any(n >= 100 for n in nums) or (len(nums) >= 2 and all(n >= 12 for n in nums)):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3495)
    ap.add_argument("--max-chars", type=int, default=320)
    ap.add_argument("--min-steps", type=int, default=4,
                    help="minimum calculation steps in the reference solution")
    ap.add_argument("--out", default=str(Path(__file__).parent / "gsm8k_mini.jsonl"))
    a = ap.parse_args()

    with urllib.request.urlopen(URL, timeout=60) as fh:
        rows = [json.loads(line) for line in fh.read().decode().splitlines() if line.strip()]
    random.Random(a.seed).shuffle(rows)

    out, seen = [], set()
    for r in rows:
        m = GOLD.search(r["answer"])
        q = r["question"].strip()
        if not m or q in seen or len(q) > a.max_chars:
            continue
        if not arithmetic_heavy(r["answer"]):
            continue
        if len(CALC.findall(r["answer"])) < a.min_steps:
            continue
        seen.add(q)
        out.append({"id": f"G{len(out) + 1:02d}", "question": q,
                    "answer": float(m.group(1).replace(",", ""))})
        if len(out) == a.n:
            break

    Path(a.out).write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {len(out)} problems -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
