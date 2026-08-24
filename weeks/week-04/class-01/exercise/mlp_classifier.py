#!/usr/bin/env python3
"""W4C1 starter, an MLP text classifier in PyTorch.

Input recipe: tokenize -> look up embeddings -> AVERAGE them -> feed an MLP.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step1 -q

When all three steps are done, the demo runs:
    python weeks/week-04/class-01/exercise/mlp_classifier.py

Everything is tiny and CPU-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

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
    """Average the embedding vectors of a document's tokens -> a single vector.

    Returns a 1-D tensor of shape (embedding_dim,).
    """
    ids = [vocab.get(tok, 0) for tok in tokenize(text)]
    if not ids:
        return torch.zeros(emb.embedding_dim)
    idx = torch.tensor(ids, dtype=torch.long)
    vectors = emb(idx)  # shape (num_tokens, embedding_dim)
    # TODO (STEP 1): implement. Check with: pytest -k step1
    # Collapse the token axis: vectors.mean(dim=0)
    raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (STEP 2): implement. Check with: pytest -k step2
        # Return raw logits: no softmax (CrossEntropyLoss applies its own).
        raise NotImplementedError


def train(
    model: MLP,
    emb: nn.Embedding,
    vocab: dict[str, int],
    data: list[tuple[str, int]],
    epochs: int = 200,
    lr: float = 0.05,
) -> list[float]:
    """Train the model + embeddings jointly. Returns the per-epoch loss."""
    params = list(model.parameters()) + list(emb.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.tensor([label for _, label in data], dtype=torch.long)

    history: list[float] = []
    for _ in range(epochs):
        # Re-embed each step so gradients also reach the embedding table.
        X = torch.stack([embed_document(t, vocab, emb) for t, _ in data])
        # TODO (STEP 3): implement. Check with: pytest -k step3
        #   1) optimizer.zero_grad()
        #   2) logits = model(X)
        #   3) loss = loss_fn(logits, y)
        #   4) loss.backward()
        #   5) optimizer.step()
        raise NotImplementedError
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
    raise SystemExit(main())
