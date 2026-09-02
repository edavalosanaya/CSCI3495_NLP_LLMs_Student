#!/usr/bin/env python3
"""W4C2 reference solution, a character-level RNN that invents names.

# TEST_SWEEP
Run inside the course container:
    python weeks/week-04/class-02/solutions/char_rnn.py
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

SEED = 1  # default: same names for everyone; override with --seed
END = "."

NAMES = [
    "tyrannosaurus", "triceratops", "stegosaurus", "velociraptor", "brachiosaurus",
    "allosaurus", "diplodocus", "ankylosaurus", "spinosaurus", "iguanodon",
    "apatosaurus", "brontosaurus", "pterodactyl", "archaeopteryx", "compsognathus",
    "deinonychus", "gallimimus", "parasaurolophus", "carnotaurus", "dilophosaurus",
    "megalosaurus", "ornithomimus", "pachycephalosaurus", "therizinosaurus",
    "albertosaurus", "baryonyx", "ceratosaurus", "gigantosaurus", "maiasaura",
    "oviraptor", "protoceratops", "saurolophus", "styracosaurus", "utahraptor",
]


def build_vocab(names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(names)) | {END})
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 24, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, ids: torch.Tensor, h0: torch.Tensor | None = None):
        x = self.emb(ids)
        out, h_n = self.rnn(x, h0)
        logits = self.out(out)
        return logits, h_n


def make_training_pairs(name: str, stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    in_chars = list(name)
    out_chars = list(name[1:]) + [END]
    xin = torch.tensor([stoi[c] for c in in_chars], dtype=torch.long)
    yt = torch.tensor([stoi[c] for c in out_chars], dtype=torch.long)
    return xin, yt


@torch.no_grad()
def sample(model: CharRNN, stoi, itos, seed: str = "t", max_len: int = 20) -> str:
    model.eval()
    result = list(seed)
    seed_ids = []
    for c in seed:
        seed_ids.append(stoi[c])
    ids = torch.tensor([seed_ids], dtype=torch.long)
    logits, h = model(ids)
    for _ in range(max_len):
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1))
        ch = itos[nxt]
        if ch == END:
            break
        result.append(ch)
        ids = torch.tensor([[nxt]], dtype=torch.long)
        logits, h = model(ids, h)
    return "".join(result)


def train(model, names, stoi, epochs: int = 400, lr: float = 0.01) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    pairs = [make_training_pairs(n, stoi) for n in names]
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.zeros(())
        for xin, yt in pairs:
            logits, _ = model(xin.unsqueeze(0))
            loss = loss + loss_fn(logits.squeeze(0), yt)
        loss = loss / len(pairs)
        loss.backward()
        opt.step()
        history.append(float(loss))
    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED,
                    help="RNG seed; default is fixed so runs are reproducible, "
                         "pass your own for unique names")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    stoi, itos = build_vocab(NAMES)
    model = CharRNN(len(stoi))
    history = train(model, NAMES, stoi)
    print(f"vocab size: {len(stoi)}")
    print(f"epoch   1 loss: {history[0]:.4f}")
    print(f"epoch {len(history):>3} loss: {history[-1]:.4f}")
    print("\nGenerated dinosaur names:")
    generated = [sample(model, stoi, itos, seed=s) for s in ["t", "a", "s", "v", "m", "r"]]
    for g in generated:
        print("  " + g)

    # sanity checks for the test sweep
    assert history[-1] < history[0], "loss should decrease"
    assert all(len(g) >= 2 for g in generated), "names should be non-trivial"
    assert any(g not in NAMES for g in generated), "should generate novel names"
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
