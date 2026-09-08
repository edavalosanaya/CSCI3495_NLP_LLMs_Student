#!/usr/bin/env python3
"""Download the small data files the labs need. Run once after `uv sync`.

    uv run python scripts/setup_data.py

The old Docker image baked these into the layer. A uv environment pins Python
packages, not data, so this is the one thing that has to be fetched separately.
Everything here is a few MB and is cached in ~/nltk_data afterwards.
"""
import sys

NLTK_SETS = ["punkt", "punkt_tab", "stopwords", "wordnet",
             # 2000 labelled film reviews (Pang & Lee 2004), the corpus the
             # W6C1 competition is scored on. Distributed with NLTK by
             # permission, so we fetch it rather than redistribute it.
             "movie_reviews"]

# Small Hugging Face models used from Week 7 on. Fetched once, then cached in
# ~/.cache/huggingface. Together they are well under 500 MB; the course never
# uses a model big enough to need a GPU.
HF_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence embeddings, W13C2"),
    ("distilbert-base-uncased", "masked language modelling, W7C1 and W8C2"),
    ("distilgpt2", "text generation and decoding, W9C1"),
]


def main() -> int:
    try:
        import nltk
    except ImportError:
        print("nltk is not installed. Run `uv sync` first.")
        return 1

    ok = True
    for name in NLTK_SETS:
        got = nltk.download(name, quiet=True)
        print(f"  {'[ok] ' if got else '[FAIL]'} {name}")
        ok = ok and got

    print()
    for name, why in HF_MODELS:
        try:
            from transformers import AutoTokenizer, AutoModel
            AutoTokenizer.from_pretrained(name)
            AutoModel.from_pretrained(name)
            print(f"  [ok]  {name}  ({why})")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}")
            ok = False

    if ok:
        print("\nData ready. Next:  uv run jupyter lab")
        return 0
    print("\nSome downloads failed. Check your network and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
