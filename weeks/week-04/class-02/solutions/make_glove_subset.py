#!/usr/bin/env python3
"""Trim real GloVe vectors down to the file the W4C2 lab ships.

GloVe 6B 50d (Pennington, Socher & Manning, EMNLP 2014), trained on Wikipedia
2014 + Gigaword 5 and released under the PDDL v1.0, so a trimmed copy may be
redistributed. The full file is 171 MB and 400k words; we keep the 20k most
frequent alphabetic ones, which is 3.8 MB.
"""
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path

KEEP = 20_000
SOURCE = ("antokun/glove.6B.50d", "glove.6B.50d.txt")


def build():
    path = hf_hub_download(SOURCE[0], SOURCE[1], repo_type="dataset")
    words, vectors = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            word = parts[0]
            # GloVe's file is ordered by frequency, so taking the first KEEP
            # survivors takes the most common words. Punctuation, numerals and
            # one-letter fragments make nearest-neighbour lists unreadable.
            if not word.isalpha() or len(word) > 20:
                continue
            if len(word) == 1 and word not in ("a", "i"):
                continue
            words.append(word)
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
            if len(words) == KEEP:
                break
    return np.array(words), np.vstack(vectors)


def main() -> None:
    words, vectors = build()
    out = Path(__file__).resolve().parents[1] / "exercise" / "data" / "glove-50d-20k.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, words=words, vectors=vectors)
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  {len(words)} words x {vectors.shape[1]} dims  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
