"""W1C2 reference solution, text processing tools."""
from __future__ import annotations
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"(?<!\w)@(\w+)")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def extract(text: str) -> dict[str, list[str]]:
    return {
        "emails": _EMAIL_RE.findall(text),
        "urls": _URL_RE.findall(text),
        "mentions": _MENTION_RE.findall(text),
    }


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    # D[i][j] = edit distance between a[:i] and b[:j]
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost, # substitution
            )
    return d[m][n]
