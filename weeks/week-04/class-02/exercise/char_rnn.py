#!/usr/bin/env python3
"""W4C2 starter, a character-level RNN that invents names."""
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
        """GIVEN. (logits, h_n) for a batch of character ids.

        logits is (batch, seq_len, vocab_size), one raw score per vocabulary
        character at every position. h_n is the final hidden state, which
        sample feeds back in on the next call.
        """
        x = self.emb(ids)
        out, h_n = self.rnn(x, h0)
        logits = self.out(out)
        return logits, h_n


def rnn_step(h_prev: torch.Tensor, x_t: torch.Tensor,
             w_h: torch.Tensor, w_x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One step of the recurrence: the new hidden state.

    Args:
        h_prev: shape (hidden,). The hidden state carried in from the previous
            character, or zeros at the start of a name.
        x_t: shape (emb_dim,). The current character, already embedded.
        w_h: shape (hidden, hidden). Applied to h_prev.
        w_x: shape (hidden, emb_dim). Applied to x_t.
        b: shape (hidden,). Added once, after both products.

    Returns:
        Shape (hidden,), the new hidden state. Every entry is between -1 and
        1, because the activation g in the formula is tanh.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   This is the formula in README section 3, one line.
    #   Use torch.tanh, and @ for a matrix times a vector.
    #
    raise NotImplementedError


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
    """GIVEN. Run one character through nn.RNN and through your rnn_step.

    Returns both hidden states. If your rnn_step is right they agree to
    floating-point noise, which is the point: nn.RNN is the formula.
    """
    ids = torch.tensor([[stoi[char]]], dtype=torch.long)
    x = model.emb(ids)
    _, h_torch = model.rnn(x)

    w_h, w_x, b = rnn_weights(model)
    h_zero = torch.zeros(model.rnn.hidden_size)
    h_mine = rnn_step(h_zero, x[0, 0], w_h, w_x, b)
    return h_torch[0, 0], h_mine


def make_training_pairs(name: str, stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """GIVEN. (input_ids, target_ids): the name, and the name shifted left
    by one and ended with END. "abc" gives inputs for "abc", targets for "bc."
    """
    in_chars = list(name)
    out_chars = list(name[1:]) + [END]
    xin = torch.tensor([stoi[c] for c in in_chars], dtype=torch.long)
    yt = torch.tensor([stoi[c] for c in out_chars], dtype=torch.long)
    return xin, yt


def sample_next(logits: torch.Tensor) -> int:
    """Draw one character index from the scores at the last position.

    Args:
        logits: shape (1, seq_len, vocab_size), straight from the model. Only
            the last position matters: it holds the scores for the character
            that comes next. The scores are raw, so they are free to be
            negative and do not sum to 1.

    Returns:
        One vocabulary index, as a plain int. Two calls on the same logits are
        free to return different characters, and normally will.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   This is the formula in README section 4, then a draw.
    #
    #   take the scores at the last position, logits[0, -1]
    #   turn them into probabilities with torch.softmax
    #   draw ONE index from that distribution with torch.multinomial
    #   return it as an int
    #
    #   Draw, do not take the highest-scoring character, or every name this
    #   model generates will be the same name.
    #
    raise NotImplementedError


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
        total = 0.0
        opt.zero_grad()
        loss = torch.zeros(())
        for xin, yt in pairs:
            logits, _ = model(xin.unsqueeze(0))           # (1, seq, vocab)
            loss = loss + loss_fn(logits.squeeze(0), yt)
        loss = loss / len(pairs)
        loss.backward()
        opt.step()
        total = float(loss.detach())
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
    h_torch, h_mine = compare_one_step(model, stoi)
    print("one step of the recurrence, nn.RNN vs your rnn_step:")
    print(f"  nn.RNN   {h_torch[:4].tolist()}")
    print(f"  rnn_step {h_mine[:4].tolist()}")
    print(f"  max difference: {float((h_torch - h_mine).abs().max()):.2e}")

    print(f"\nvocab size: {len(stoi)}")
    print(f"epoch   1 loss: {history[0]:.4f}")
    print(f"epoch {len(history):>3} loss: {history[-1]:.4f}")
    print("\nGenerated dinosaur names:")
    for seed in ["t", "a", "s", "v", "m", "r"]:
        print("  " + sample(model, stoi, itos, seed=seed))
    return 0


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        raise SystemExit(main())
    except NotImplementedError:
        print("char_rnn.py is not finished yet: fill in the next TODO in this file, then re-run.")
