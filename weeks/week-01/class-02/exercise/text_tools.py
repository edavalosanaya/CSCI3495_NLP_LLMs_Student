"""W1C2 starter: text processing tools.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step1 -q

When all four steps are done, the whole suite is green:
    python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -q
"""
from __future__ import annotations
import re

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"(?<!\w)@(\w+)")


def normalize(text: str) -> str:
    """Lowercase, collapse runs of whitespace to one space, strip ends."""
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    """Split normalized text into word tokens (letters/digits and standalone punctuation)."""
    # TODO (STEP 2): implement. Check with: pytest -k step2
    # Hint: normalize first, then re.findall with r"\w+|[^\w\s]"
    raise NotImplementedError


def extract(text: str) -> dict[str, list[str]]:
    """Return {'emails': [...], 'urls': [...], 'mentions': [...]} found via regex."""
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    return {
        "emails": _EMAIL_RE.findall(text),
        "urls": _URL_RE.findall(text),
        "mentions": _MENTION_RE.findall(text),
    }


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance via dynamic programming."""
    # TODO (STEP 4): implement the O(mn) DP table. Check with: pytest -k step4
    # D[i][j] compares a[:i] with b[:j], so compare a[i-1] and b[j-1].
    raise NotImplementedError
