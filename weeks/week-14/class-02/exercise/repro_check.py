#!/usr/bin/env python3
"""W14C2, Reproducibility quick-check (offline, no network).

A tiny pre-final sanity tool. Given a Python entry point, it checks the two
things graders look at first:

  1. Does the file import/parse cleanly inside the course Docker image?
  2. Does it *look* deterministic (does it seed the common RNGs)?

This makes NO network calls and runs nothing dangerous, it parses the source
and reports hints. It is deliberately simple and fully unit-testable.

Usage:
    python weeks/week-14/class-02/exercise/repro_check.py path/to/entrypoint.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# Calls that signal the author tried to make results deterministic.
SEED_HINTS = (
    "random.seed",
    "np.random.seed",
    "numpy.random.seed",
    "torch.manual_seed",
    "set_seed",
)


def parses_cleanly(source: str) -> bool:
    """True if the source is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def find_seed_hints(source: str) -> list[str]:
    """Return the seeding calls found in the source (substring match)."""
    return [h for h in SEED_HINTS if h in source]


def check_file(path: Path) -> dict:
    """Run the checks on a file and return a report dict."""
    if not path.exists():
        return {"ok": False, "reason": f"file not found: {path}"}
    source = path.read_text(encoding="utf-8", errors="replace")
    parses = parses_cleanly(source)
    seeds = find_seed_hints(source)
    return {
        "ok": parses,
        "parses": parses,
        "seed_hints": seeds,
        "deterministic_hint": bool(seeds),
    }


def format_report(path: Path, report: dict) -> str:
    lines = [f"Reproducibility check: {path}"]
    if not report.get("parses", False):
        lines.append("  [FAIL] file does not parse as valid Python "
                     f"({report.get('reason', 'syntax error')}).")
        return "\n".join(lines)
    lines.append("  [OK]   parses cleanly.")
    if report["deterministic_hint"]:
        lines.append(f"  [OK]   found seeding: {', '.join(report['seed_hints'])}")
    else:
        lines.append("  [WARN] no RNG seeding found, set seeds for reproducible results.")
    lines.append("  Note: run your real entry point in Docker too; this is only a static hint.")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    path = Path(argv[0])
    report = check_file(path)
    print(format_report(path, report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
