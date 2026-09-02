#!/usr/bin/env python3
"""W4C1 starter, an MLP text classifier in PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn

def _embed_batch(data, vocab, emb) -> torch.Tensor:
    return torch.stack([embed_document(t, vocab, emb) for t, _ in data])

SEED = 0

# A tiny, deliberately separable sentiment dataset (label 1 = positive, 0 = negative).
TRAIN = [
    ("i loved this movie it was great", 1),
    ("a wonderful and brilliant film", 1),
    ("great acting truly fantastic", 1),
    ("i enjoyed it so much wonderful", 1),
    ("terrible boring and awful", 0),
    ("i hated this movie it was bad", 0),
    ("awful acting truly terrible", 0),
    ("boring dull and bad", 0),
]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_vocab(corpus: list[str]) -> dict[str, int]:
    """Map each distinct token to an integer id (0 is reserved for unknown/pad)."""
    vocab: dict[str, int] = {"<unk>": 0}
    for text in corpus:
        for tok in tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def embed_document(text: str, vocab: dict[str, int], emb: nn.Embedding) -> torch.Tensor:
    """Collapse a document into ONE vector by averaging its word embeddings.

    Args:
        text: the raw document. Tokenizing it is part of this function's job.
        vocab: token -> row index in the embedding table. Index 0 is the
            unknown-word row, and any token missing from the vocab uses it.
        emb: the embedding table. Calling it on a LongTensor of ids returns
            one row per id, and emb.embedding_dim is the width of a row.

    Returns:
        A 1-D tensor of shape (embedding_dim,). A document with no tokens at
        all still has to return that shape, so return zeros rather than
        averaging an empty list.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The averaging step is the left-hand formula in README section 2.
    #
    #   turn every token into its vocab index, falling back to the unknown row
    #   if the document produced no tokens at all, hand back a vector of zeros
    #   wrap the indices in a LongTensor and look them all up in one call
    #   average the resulting rows down to a single vector
    #
    #   Averaging over the wrong axis silently gives you a vector of the wrong
    #   length: you want one number per embedding dimension.
    #
    # raise NotImplementedError
    tokens = tokenize(text)
    ids = []
    for t in tokens:
        i = 0
        if t in vocab:
            i = vocab[t]
        ids.append(i)
    ids = torch.LongTensor(ids)
    embs = emb(ids)
    import pdb; pdb.set_trace()
    summed_embs = embs.sum(axis=1)
    summed_embs = summed_embs / emb.shape[0]
    return embs



class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of document vectors to one score per class.

        Args:
            x: shape (batch, in_dim). One row per document, each row the
                averaged vector that embed_document produced.

        Returns:
            Shape (batch, out_dim): raw scores, NOT probabilities. They are
            free to be negative and need not sum to 1.
        """
        # TODO (STEP 2): implement. Check with: pytest -k step2
        #
        #   The right-hand pair of formulas in README section 2. One line.
        #
        #   send x through the first linear layer, then the activation,
        #   then the second linear layer, and return that
        #
        #   Stop there. No softmax: CrossEntropyLoss applies it itself, and
        #   applying it twice quietly flattens the gradients.
        #
        raise NotImplementedError


def train(
    model: MLP,
    emb: nn.Embedding,
    vocab: dict[str, int],
    data: list[tuple[str, int]],
    epochs: int = 200,
    lr: float = 0.05,
) -> list[float]:
    """GIVEN. Trains the classifier and the embedding table jointly.

    Returns the per-epoch loss. Re-embeds every step so gradients reach the
    embedding table too.
    """
    params = list(model.parameters()) + list(emb.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.tensor([label for _, label in data], dtype=torch.long)

    history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        # Re-embed each step so gradients flow into the embedding table too.
        X = _embed_batch(data, vocab, emb)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        history.append(float(loss))
    return history


def accuracy(model: MLP, emb: nn.Embedding, vocab, data) -> float:
    X = torch.stack([embed_document(t, vocab, emb) for t, _ in data])
    y = torch.tensor([label for _, label in data], dtype=torch.long)
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return float((preds == y).float().mean())


def main() -> int:
    torch.manual_seed(SEED)
    vocab = build_vocab([t for t, _ in TRAIN])
    emb = nn.Embedding(len(vocab), 16)
    model = MLP(in_dim=16, hidden=8)

    history = train(model, emb, vocab, TRAIN)
    print(f"vocab size: {len(vocab)}")
    print(f"epoch   1 loss: {history[0]:.4f}")
    print(f"epoch {len(history):>3} loss: {history[-1]:.4f}")
    print(f"train accuracy: {accuracy(model, emb, vocab, TRAIN):.2f}")
    return 0


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        raise SystemExit(main())
    except NotImplementedError:
        print("mlp_classifier.py is not finished yet: fill in the next TODO in this file, then re-run.")
