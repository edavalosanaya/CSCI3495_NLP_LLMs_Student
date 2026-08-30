"""W6C1 reference solution, static vs. contextual embeddings (tiny BERT)."""
from __future__ import annotations

import math

MODEL_NAME = "prajjwal1/bert-tiny"

_CACHE: dict = {}


def load_model():
    if "pair" in _CACHE:
        return _CACHE["pair"]
    try:
        import torch  # noqa: F401
        # Use the explicit BERT classes: prajjwal1/bert-tiny ships only a
        # vocab.txt (no tokenizer.json) and a config without `model_type`, so the
        # Auto* dispatch fails under transformers >= 5. BertTokenizerFast builds
        # the fast WordPiece tokenizer straight from vocab.txt.
        from transformers import BertModel, BertTokenizerFast
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "transformers/torch not available, run inside the course container."
        ) from e
    tok = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()
    _CACHE["pair"] = (tok, model)
    return tok, model


def cosine_similarity(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def _word_token_ids(tok, word: str) -> list[int]:
    """Sub-token ids for `word` alone, without special tokens."""
    return tok(word, add_special_tokens=False)["input_ids"]


def _find_positions(haystack: list[int], needle: list[int]) -> list[int]:
    """Return the indices in `haystack` covered by the first match of `needle`."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return list(range(i, i + n))
    return []


def contextual_vector(sentence: str, word: str):
    import torch

    tok, model = load_model()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    last = out.hidden_states[-1][0]  # (seq_len, hidden)

    ids = enc["input_ids"][0].tolist()
    word_ids = _word_token_ids(tok, word)
    positions = _find_positions(ids, word_ids)
    if not positions:
        # Fallback: average everything except special tokens.
        positions = list(range(1, len(ids) - 1))
    vec = last[positions].mean(dim=0)
    return vec.tolist()


def static_vector(word: str):
    import torch

    tok, model = load_model()
    emb = model.get_input_embeddings()  # nn.Embedding: id -> vector
    word_ids = _word_token_ids(tok, word)
    with torch.no_grad():
        vecs = emb(torch.tensor(word_ids))
    return vecs.mean(dim=0).tolist()


def _demo() -> None:
    pairs = [
        ("I sat by the river bank and watched the water.", "bank"),
        ("I deposited my paycheck at the bank downtown.", "bank"),
    ]
    cvecs = [contextual_vector(s, w) for s, w in pairs]
    svecs = [static_vector(w) for _, w in pairs]
    print(f"Contextual cosine('bank' river vs. money): {cosine_similarity(*cvecs):.3f}")
    print(f"Static     cosine('bank' river vs. money): {cosine_similarity(*svecs):.3f}")
    print("Expect: static == 1.000 (identical), contextual < static (sense-dependent).")


if __name__ == "__main__":
    _demo()
