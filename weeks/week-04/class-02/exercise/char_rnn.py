#!/usr/bin/env python3
"""W4C2 starter, a character-level RNN that invents names.

Train on a tiny list of dinosaur names; learn next-character prediction; then
generate brand-new names by sampling. Run inside the course container:
    python weeks/week-04/class-02/exercise/char_rnn.py

NOTE ON RANDOMNESS: by default this runs with a fixed seed (SEED = 1), so
EVERYONE in class gets the exact same names; that keeps the run reproducible
and the tests deterministic. Before the name-vote activity, make the names
YOURS by passing your own seed, e.g.:
    python char_rnn.py --seed 42        (any number you like)

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-04/class-02/exercise/test_char_rnn.py -k step1 -q

Everything is tiny and CPU-only.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

SEED = 1  # default: same names for everyone; override with --seed
END = "."  # marks the end of a name

# A small, real-ish dinosaur-name corpus (lowercased). The model learns the
# *shape* of these and produces new ones, it should not just copy them.
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
        """ids: (batch, seq_len) long. Returns (logits, h_n).

        logits: (batch, seq_len, vocab_size); h_n: final hidden state.
        """
        # TODO (STEP 1): implement. Check with: pytest -k step1
        #   1) x = self.emb(ids)                # (batch, seq, emb_dim)
        #   2) out, h_n = self.rnn(x, h0)        # out: (batch, seq, hidden)
        #   3) logits = self.out(out)           # (batch, seq, vocab_size)
        #   return logits, h_n
        raise NotImplementedError


def make_training_pairs(name: str, stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """For a name, return (input_ids, target_ids).

    Input is the name; target is the name shifted left by one, ending in END.
    Example: "abc" -> input ids for "abc", target ids for "bc."
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #   in_chars  = list(name)
    #   out_chars = list(name[1:]) + [END]
    #   convert each to ids via stoi, return as 1-D LongTensors
    raise NotImplementedError


@torch.no_grad()
def sample(model: CharRNN, stoi, itos, seed: str = "t", max_len: int = 20) -> str:
    """Generate a name autoregressively, starting from `seed`, stopping at END."""
    model.eval()
    result = list(seed)
    ids = torch.tensor([[stoi[c] for c in seed]], dtype=torch.long)
    logits, h = model(ids)
    for _ in range(max_len):
        last_logits = logits[0, -1]               # (vocab_size,)
        probs = torch.softmax(last_logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1))
        ch = itos[nxt]
        # TODO (STEP 3): implement. Check with: pytest -k step3
        #   if ch == END: break; otherwise append ch to result, then feed the
        #   new id back through the model WITH the hidden state h, reassigning
        #   both: logits, h = model(ids, h)
        raise NotImplementedError
    return "".join(result)


def train(model, names, stoi, epochs: int = 400, lr: float = 0.01) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    pairs = [make_training_pairs(n, stoi) for n in names]
    history = []
    for _ in range(epochs):
        total = 0.0
        opt.zero_grad()
        loss = torch.zeros(())
        for xin, yt in pairs:
            logits, _ = model(xin.unsqueeze(0))           # (1, seq, vocab)
            loss = loss + loss_fn(logits.squeeze(0), yt)
        loss = loss / len(pairs)
        loss.backward()
        opt.step()
        total = float(loss)
        history.append(total)
    return history


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="RNG seed; the default gives everyone identical names, "
                         "pass your own (any number) to get names nobody else has")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.seed == SEED:
        print(f"(seed {SEED}: same names as everyone else; rerun with --seed <n> for your own)")
    stoi, itos = build_vocab(NAMES)
    model = CharRNN(len(stoi))
    history = train(model, NAMES, stoi)
    print(f"vocab size: {len(stoi)}")
    print(f"epoch   1 loss: {history[0]:.4f}")
    print(f"epoch {len(history):>3} loss: {history[-1]:.4f}")
    print("\nGenerated dinosaur names:")
    for seed in ["t", "a", "s", "v", "m", "r"]:
        print("  " + sample(model, stoi, itos, seed=seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
