"""W1C2 starter: text processing tools.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step1 -q

When all four steps are done, the whole suite is green:
    python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -q
"""
from __future__ import annotations
import re


def normalize(text: str) -> str:
    """Lowercase, collapse runs of whitespace to one space, strip ends."""
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens (letters/digits and standalone punctuation)."""
    # TODO (STEP 2): implement. Check with: pytest -k step2
    # Hint: normalize first, then re.findall with r"\w+|[^\w\s]"
    raise NotImplementedError


def extract(text: str) -> dict[str, list[str]]:
    """Return {'emails': [...], 'urls': [...], 'mentions': [...]} found via regex."""
    # TODO (STEP 3): implement. Check with: pytest -k step3
    # Use the RAW text (not normalized). Careful: a naive @-pattern also
    # matches inside an email address.
    raise NotImplementedError


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance via dynamic programming."""
    # TODO (STEP 4): implement the O(mn) DP table. Check with: pytest -k step4
    # D[i][j] compares a[:i] with b[:j], so compare a[i-1] and b[j-1].
    raise NotImplementedError
