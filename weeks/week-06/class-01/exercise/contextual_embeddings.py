"""W6C1 starter, static vs. contextual embeddings.

Goal: show that a *static* embedding gives the SAME vector for a word in any
sentence, while a *contextual* model (a tiny BERT) gives DIFFERENT vectors
depending on the sentence.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-06/class-01/exercise/test_contextual.py -k step1 -q

Everything runs CPU-only with a TINY model (prajjwal1/bert-tiny). The first run
downloads a few MB; after that it is cached. If transformers/torch are missing,
the helpers raise a clear error.
"""
from __future__ import annotations

import math

MODEL_NAME = "prajjwal1/bert-tiny"


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
    """Return the contextual embedding (a list of floats) of `word` as it
    appears in `sentence`, using a tiny BERT model.

    Steps (hint):
      1. Tokenize `sentence` with the model's tokenizer (return_tensors="pt").
      2. Run the model with output_hidden_states; take the LAST hidden layer.
      3. Find the token position(s) for `word` and average their vectors.
         (A simple approach: tokenize the lone word to get its sub-token ids,
          then find where they occur in the sentence's input_ids.)
      4. Return the resulting vector as a plain python list of floats.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    raise NotImplementedError


def static_vector(word: str):
    """Return a *static* embedding of `word`: the model's input (word-piece)
    embedding, which does NOT depend on any sentence.

    Hint: model.get_input_embeddings() maps token ids -> vectors. Average the
    sub-token embeddings for `word`. This is the same regardless of context.
    """
    # TODO (STEP 3): implement. Check with: pytest -k step3
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
    _demo()
