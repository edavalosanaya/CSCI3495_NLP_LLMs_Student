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
    vocab: Counter = Counter()
    for line in corpus:
        for word in line.lower().split():
            symbols = tuple(word) + (END,)
            vocab[symbols] += 1
    return dict(vocab)


def count_pairs(vocab: dict[tuple[str, ...], int]) -> Counter:
    """Count every adjacent pair of symbols, weighted by how common the word is.

    Args:
        vocab: symbol tuple -> how many times that word appears in the corpus.
            A word is a tuple like ("l", "o", "w", "</w>"), and it gets shorter
            as merges are applied.

    Returns:
        A Counter of (left symbol, right symbol) -> total count across the
        whole corpus.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   for each word and its frequency:
    #       walk its symbols in overlapping neighbouring pairs
    #       add that word's FREQUENCY to each pair's running count
    #
    #   Add the frequency, not 1. A word that appears five times contributes
    #   five to each of its pairs, and getting this wrong changes which pair
    #   wins the very first merge.
    #
    raise NotImplementedError


def merge_pair(
    pair: tuple[str, str], vocab: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """Fuse one pair of symbols everywhere it occurs, giving a new vocab.

    Args:
        pair: the two symbols to glue together, e.g. ("e", "r").
        vocab: the current symbol tuple -> frequency mapping.

    Returns:
        A NEW mapping with the pair replaced by the single joined symbol
        wherever the two were adjacent. Merging ("e", "r") turns
        ("l", "o", "w", "e", "r", "</w>") into ("l", "o", "w", "er", "</w>").
        The input is not modified.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   for each word, scan its symbols left to right:
    #       when the next two symbols are exactly this pair, emit them joined
    #           as one symbol and skip past BOTH
    #       otherwise emit the current symbol and move on by one
    #   put the rewritten word in the new mapping, ADDING its frequency to
    #       whatever is already there
    #
    #   That last point matters: two different words can collapse to the same
    #   symbols, and overwriting instead of adding silently loses a word.
    #
    raise NotImplementedError


def train_bpe(
    corpus: list[str], num_merges: int
) -> list[tuple[str, str]]:
    """Learn a merge list by repeatedly fusing the most frequent pair.

    Args:
        corpus: the training sentences.
        num_merges: the maximum number of merges to learn. Fewer are returned
            if the corpus runs out of adjacent pairs first.

    Returns:
        The merges in the order they were learned. Order is the whole point:
        encoding replays this list from the start, so a later merge can only
        build on symbols an earlier one created.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
    #
    #   The loop that drives the two functions above. build_vocab is given.
    #
    #   build the starting vocab from the corpus
    #   up to num_merges times:
    #       count the pairs; if there are none left, stop early
    #       pick the most frequent pair, breaking ties by the pair itself so
    #           two runs on the same corpus always agree
    #       merge it, and record it
    #   return the merges in order
    #
    raise NotImplementedError


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """GIVEN. Tokenizes one word by replaying the learned merges in order."""
    symbols: list[str] = list(word.lower()) + [END]
    for a, b in merges:
        merged = a + b
        out: list[str] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        demo = ["low low low low low", "lower lower", "newest newest newest", "widest"]
        merges = train_bpe(demo, num_merges=10)
        print("Learned merges (in order):")
        for i, m in enumerate(merges, 1):
            print(f"  {i:2d}. {m[0]!r} + {m[1]!r}")
        print("\nEncoding 'lowest':", encode_word("lowest", merges))
    except NotImplementedError:
        print("bpe.py is not finished yet: fill in the next TODO in this file, then re-run.")
