"""W6C1 starter, static vs. contextual embeddings."""
from __future__ import annotations

import math

MODEL_NAME = "prajjwal1/bert-tiny"


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


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    cos(u, v) = (u . v) / (||u|| * ||v||)
    """
    # GIVEN (STEP 1): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def contextual_vector(sentence: str, word: str):
    """The vector a word gets FROM the sentence it sits in.

    Args:
        sentence: the full sentence, which the model reads in one pass.
        word: the word to pull out of it. It may split into several
            word-pieces, and it may appear more than once.

    Returns:
        A plain list of floats, one per hidden dimension. The same word in a
        different sentence must give a different list: that difference is the
        entire point of the lab.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The formula is in README section 3.
    #
    #   load the tokenizer and model with the given helper
    #   encode the sentence as pytorch tensors and run the model, asking it to
    #       return its hidden states
    #   take the LAST hidden layer, for this one sentence
    #   work out which token positions the word occupies: _word_token_ids gives
    #       the word's sub-token ids, _find_positions locates them in the
    #       sentence's ids. Both are written for you
    #   average the vectors at those positions and return them as a plain list
    #
    #   Run the model under torch.no_grad(): nothing here is training.
    #
    raise NotImplementedError


def static_vector(word: str):
    """The vector a word gets on its own, with no sentence at all.

    Args:
        word: the word to look up. Note there is no sentence parameter, and
            that absence is the comparison the lab is built around.

    Returns:
        A plain list of floats, the same list every time for a given word.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The first formula in README section 4.
    #
    #   load the tokenizer and model with the given helper
    #   ask the model for its input embedding table, which maps a token id
    #       straight to a vector without running the network at all
    #   get the word's sub-token ids with _word_token_ids
    #   look those ids up in the table and average them
    #
    #   Nothing here touches a sentence, and nothing here runs a forward pass.
    #
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Provided helper: load (and cache) the tiny model + tokenizer.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def load_model():
    """Load tokenizer + model once and cache them. Raises a clear error if the
    libraries or the model are unavailable."""
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


def _demo() -> None:
    pairs = [
        ("I sat by the river bank and watched the water.", "bank"),
        ("I deposited my paycheck at the bank downtown.", "bank"),
    ]
    cvecs = [contextual_vector(s, w) for s, w in pairs]
    svecs = [static_vector(w) for _, w in pairs]
    print(f"Contextual cosine('bank' river vs. money): {cosine_similarity(*cvecs):.3f}")
    print(f"Static     cosine('bank' river vs. money): {cosine_similarity(*svecs):.3f}")
    print("Expect: static ≈ 1.000 (identical), contextual < static (sense-dependent).")


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        _demo()
    except NotImplementedError:
        print("contextual_embeddings.py is not finished yet: fill in the next TODO in this file, then re-run.")
