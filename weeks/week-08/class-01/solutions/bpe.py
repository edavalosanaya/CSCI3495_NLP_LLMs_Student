"""W8C1 reference solution, Byte-Pair Encoding (BPE) tokenizer from scratch."""
from __future__ import annotations

from collections import Counter

END = "</w>"


def build_vocab(corpus: list[str]) -> dict[tuple[str, ...], int]:
    vocab: Counter = Counter()
    for line in corpus:
        for word in line.lower().split():
            symbols = tuple(word) + (END,)
            vocab[symbols] += 1
    return dict(vocab)


def count_pairs(vocab: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(
    pair: tuple[str, str], vocab: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    a, b = pair
    merged = a + b
    new_vocab: dict[tuple[str, ...], int] = {}
    for symbols, freq in vocab.items():
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
        new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + freq
    return new_vocab


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    vocab = build_vocab(corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = count_pairs(vocab)
        if not pairs:
            break
        # Most frequent pair; deterministic tie-break by the pair itself.
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
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
    demo = ["low low low low low", "lower lower", "newest newest newest", "widest"]
    merges = train_bpe(demo, num_merges=10)
    print("Learned merges (in order):")
    for i, m in enumerate(merges, 1):
        print(f"  {i:2d}. {m[0]!r} + {m[1]!r}")
    print("\nEncoding 'lowest':", encode_word("lowest", merges))
