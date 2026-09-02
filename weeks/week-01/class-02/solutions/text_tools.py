"""W1C2 reference solution, text processing tools."""
from __future__ import annotations
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"(?<!\w)@(\w+)")

SAMPLE = "  Email A.B+x@Mail.co  or ping @alice at https://x.io/p?q=1 !! "


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
            # Row i is about the character a[i - 1], not a[i].
            if a[i - 1] == b[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            delete = d[i - 1][j] + 1
            insert = d[i][j - 1] + 1
            substitute = d[i - 1][j - 1] + substitution_cost

            best = delete
            if insert < best:
                best = insert
            if substitute < best:
                best = substitute

            d[i][j] = best

    return d[m][n]


def _demo() -> None:
    print("=" * 60)
    print("Text tools")
    print("=" * 60)
    print(f"  raw        {SAMPLE!r}")
    print(f"  normalized {normalize(SAMPLE)!r}")
    print(f"  tokens     {tokenize(SAMPLE)}")
    for k, v in extract(SAMPLE).items():
        print(f"  {k:<10} {v}")
    print()
    for a, b in [("intention", "execution"), ("kitten", "sitting"), ("same", "same")]:
        print(f"  edit_distance({a!r}, {b!r}) = {edit_distance(a, b)}")


if __name__ == "__main__":
    _demo()
