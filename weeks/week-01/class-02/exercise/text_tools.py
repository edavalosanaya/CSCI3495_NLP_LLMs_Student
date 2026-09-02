#!/usr/bin/env python3
"""W1C2 starter, text processing tools. See README.md."""
from __future__ import annotations
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"(?<!\w)@(\w+)")

SAMPLE = "  Email A.B+x@Mail.co  or ping @alice at https://x.io/p?q=1 !! "


def normalize(text: str) -> str:
    """GIVEN. Lowercase, collapse whitespace runs to one space, strip the ends."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract(text: str) -> dict[str, list[str]]:
    """GIVEN. Pull emails, URLs and @mentions out of RAW text with the regexes above."""
    return {
        "emails": _EMAIL_RE.findall(text),
        "urls": _URL_RE.findall(text),
        "mentions": _MENTION_RE.findall(text),
    }


def tokenize(text: str) -> list[str]:
    """Split text into tokens: words, and each punctuation mark on its own.

    Args:
        text: raw text, not yet cleaned. Normalizing is part of this function's
            job, so callers can hand it anything.

    Returns:
        A list of token strings, lowercased, in the order they appear.
        "Hello, world!" gives four tokens, not two: punctuation counts.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   clean the text with the helper above, then let _TOKEN_RE find every
    #   match in the cleaned string and return them as a list.
    #
    #   _TOKEN_RE already means "a run of word characters, OR one lone
    #   punctuation mark" -- you do not need to write a regex.
    #
    raise NotImplementedError


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest single-character edits turning a into b.

    An edit is one insertion, one deletion, or one substitution, each costing 1.

    Args:
        a: the string being edited from. May be empty.
        b: the string being edited to. May be empty.

    Returns:
        A non-negative int. 0 when the strings are equal, and len(b) when a is
        empty, since every character has to be inserted.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The recurrence is in README section 2. Build it as a table with one row
    #   per prefix of a and one column per prefix of b.
    #
    #   fill the first column with 0, 1, 2, ...   (deleting all of a)
    #   fill the first row with 0, 1, 2, ...      (inserting all of b)
    #   for every remaining cell, in row order:
    #       substituting costs nothing when the two characters match
    #       take the cheapest of the three neighbours the recurrence names
    #   the answer is the bottom-right cell
    #
    #   Row i is about the character a[i - 1], not a[i]. That off-by-one is
    #   where this goes wrong.
    #
    raise NotImplementedError


def _demo() -> None:
    """GIVEN. Runs all four tools over SAMPLE."""
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
    try:
        _demo()
    except NotImplementedError:
        print("text_tools.py is not finished yet: fill in the next TODO in this file, then re-run.")
