"""W8C1 starter, train a Byte-Pair Encoding (BPE) tokenizer from scratch.

No network, no libraries beyond the standard library.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-08/class-01/exercise/test_bpe.py -k step1 -q

Representation we use throughout:
- A "word" is split into a tuple of symbols, e.g. ("l", "o", "w", "</w>").
- We append the end-of-word marker "</w>" so the tokenizer knows where words
  end (and so "er" at a word end differs from "er" inside a word).
- A "vocab" is a dict {word_tuple: frequency} over the training corpus.
"""
from __future__ import annotations

from collections import Counter

END = "</w>"


def build_vocab(corpus: list[str]) -> dict[tuple[str, ...], int]:
    """Turn a list of sentences into {symbol_tuple: count} over whitespace words.

    Lowercase, split on whitespace, and represent each word as a tuple of its
    characters plus the END marker. Example:
        "low low"  ->  {("l","o","w","</w>"): 2}
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


def count_pairs(vocab: dict[tuple[str, ...], int]) -> Counter:
    """Count adjacent symbol pairs across the vocab, weighted by word frequency.

    Return a Counter mapping (sym_a, sym_b) -> total count.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


def merge_pair(
    pair: tuple[str, str], vocab: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """Return a new vocab where every adjacent occurrence of `pair` is merged.

    Merging ("e","r") in ("l","o","w","e","r","</w>") gives
    ("l","o","w","er","</w>").
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    raise NotImplementedError


def train_bpe(
    corpus: list[str], num_merges: int
) -> list[tuple[str, str]]:
    """Run BPE training for `num_merges` steps.

    Repeatedly: count pairs, pick the most frequent (break ties by the pair's
    sort order for determinism), merge it. Return the ordered list of merges.
    Stop early if there are no pairs left to merge.
    """
    # TODO (STEP 4): implement. Check with: pytest -k step4
    raise NotImplementedError


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize a single word by applying the learned merges in order.

    Start from characters + END, then for each learned merge (in order) fuse
    all adjacent occurrences of that pair. Returns the final list of subwords.
    """
    # TODO (STEP 5): implement. Check with: pytest -k step5
    raise NotImplementedError


if __name__ == "__main__":
    demo = ["low low low low low", "lower lower", "newest newest newest", "widest"]
    merges = train_bpe(demo, num_merges=10)
    print("Learned merges (in order):")
    for i, m in enumerate(merges, 1):
        print(f"  {i:2d}. {m[0]!r} + {m[1]!r}")
    print("\nEncoding 'lowest':", encode_word("lowest", merges))
