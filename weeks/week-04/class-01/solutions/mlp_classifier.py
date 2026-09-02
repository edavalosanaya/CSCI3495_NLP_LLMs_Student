#!/usr/bin/env python3
"""W4C1 reference solution, an MLP text classifier in PyTorch.

# TEST_SWEEP
Run inside the course container:
    python weeks/week-04/class-01/solutions/mlp_classifier.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

SEED = 0

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
    vocab: dict[str, int] = {"<unk>": 0}
    for text in corpus:
        for tok in tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def embed_document(text: str, vocab: dict[str, int], emb: nn.Embedding) -> torch.Tensor:
    ids = []
    for tok in tokenize(text):
        # Row 0 is the unknown-word row, used for any token not in the vocab.
        ids.append(vocab.get(tok, 0))

    if len(ids) == 0:
        # No tokens at all, but the caller still needs a vector of the right
        # width, so hand back zeros rather than averaging an empty list.
        return torch.zeros(emb.embedding_dim)

    idx = torch.tensor(ids, dtype=torch.long)
    vectors = emb(idx)              # (num_tokens, embedding_dim)
    return vectors.mean(dim=0)      # average down to (embedding_dim,)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _embed_batch(data, vocab, emb) -> torch.Tensor:
    return torch.stack([embed_document(t, vocab, emb) for t, _ in data])


def train(
    model: MLP,
    emb: nn.Embedding,
    vocab: dict[str, int],
    data: list[tuple[str, int]],
    epochs: int = 200,
    lr: float = 0.05,
) -> list[float]:
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
    y = torch.tensor([label for _, label in data], dtype=torch.long)
    with torch.no_grad():
        preds = model(_embed_batch(data, vocab, emb)).argmax(dim=1)
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
    acc = accuracy(model, emb, vocab, TRAIN)
    print(f"train accuracy: {acc:.2f}")
    assert history[-1] < history[0], "loss should decrease"
    assert acc == 1.0, "should fit this tiny separable set"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
