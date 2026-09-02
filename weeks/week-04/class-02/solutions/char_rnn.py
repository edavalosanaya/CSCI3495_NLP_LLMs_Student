#!/usr/bin/env python3
"""W4C2 reference solution, a character-level RNN that invents names."""
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
        """GIVEN. (logits, h_n) for a batch of character ids."""
        x = self.emb(ids)
        out, h_n = self.rnn(x, h0)
        logits = self.out(out)
        return logits, h_n



def rnn_step(h_prev: torch.Tensor, x_t: torch.Tensor,
             w_h: torch.Tensor, w_x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    combined = w_h @ h_prev + w_x @ x_t + b
    return torch.tanh(combined)


def rnn_weights(model: CharRNN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GIVEN. The three weights of the recurrence, pulled out of nn.RNN.

    nn.RNN carries two bias vectors where the formula has one, so they are
    added together here.
    """
    rnn = model.rnn
    b = rnn.bias_ih_l0 + rnn.bias_hh_l0
    return rnn.weight_hh_l0, rnn.weight_ih_l0, b


@torch.no_grad()
def compare_one_step(model: CharRNN, stoi: dict[str, int], char: str = "t"):
    """GIVEN. Run one character through nn.RNN and through rnn_step.

    Returns both hidden states, which should agree to floating-point noise.
    """
    ids = torch.tensor([[stoi[char]]], dtype=torch.long)
    x = model.emb(ids)
    _, h_torch = model.rnn(x)

    w_h, w_x, b = rnn_weights(model)
    h_zero = torch.zeros(model.rnn.hidden_size)
    h_mine = rnn_step(h_zero, x[0, 0], w_h, w_x, b)
    return h_torch[0, 0], h_mine


def make_training_pairs(name: str, stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    in_chars = list(name)
    out_chars = list(name[1:]) + [END]
    xin = torch.tensor([stoi[c] for c in in_chars], dtype=torch.long)
    yt = torch.tensor([stoi[c] for c in out_chars], dtype=torch.long)
    return xin, yt


def sample_next(logits: torch.Tensor) -> int:
    last_scores = logits[0, -1]
    probs = torch.softmax(last_scores, dim=-1)
    drawn = torch.multinomial(probs, num_samples=1)
    return int(drawn)


@torch.no_grad()
def sample(model: CharRNN, stoi, itos, seed: str = "t", max_len: int = 20) -> str:
    """GIVEN. Generate one name, asking sample_next for each character."""
    model.eval()
    result = list(seed)

    seed_ids = []
    for c in seed:
        seed_ids.append(stoi[c])
    ids = torch.tensor([seed_ids], dtype=torch.long)
    logits, h = model(ids)

    for _ in range(max_len):
        nxt = sample_next(logits)
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
        history.append(float(loss.detach()))
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
    h_torch, h_mine = compare_one_step(model, stoi)
    print(f"one step of the recurrence, nn.RNN vs your rnn_step:")
    print(f"  nn.RNN   {h_torch[:4].tolist()}")
    print(f"  rnn_step {h_mine[:4].tolist()}")
    print(f"  max difference: {float((h_torch - h_mine).abs().max()):.2e}")

    print(f"\nvocab size: {len(stoi)}")
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
