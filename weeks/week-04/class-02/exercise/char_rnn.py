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
        """Score every next-character option at every position in the sequence.

        Args:
            ids: shape (batch, seq_len), character indices.
            h0: the hidden state to start from, or None to start from zeros.
                Sampling passes the previous call's state back in here, which
                is what lets the model carry context across calls.

        Returns:
            (logits, h_n). logits is (batch, seq_len, vocab_size): one score
            per vocabulary character at every position, raw, no softmax. h_n
            is the final hidden state, to be fed back in on the next call.
        """
        # TODO (STEP 1): implement. Check with: pytest -k step1
        #
        #   The recurrence is in README section 2. Three of the layers built
        #   in __init__ do all the work, in the order they were defined.
        #
        #   turn the ids into vectors with the embedding layer
        #   run those through the RNN, handing it the incoming hidden state
        #   project the RNN's output to one score per vocabulary character
        #   return the scores AND the final hidden state, in that order
        #
        raise NotImplementedError


def make_training_pairs(name: str, stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """GIVEN. (input_ids, target_ids): the name, and the name shifted left
    by one and ended with END. "abc" gives inputs for "abc", targets for "bc."
    """
    in_chars = list(name)
    out_chars = list(name[1:]) + [END]
    xin = torch.tensor([stoi[c] for c in in_chars], dtype=torch.long)
    yt = torch.tensor([stoi[c] for c in out_chars], dtype=torch.long)
    return xin, yt


@torch.no_grad()
def sample(model: CharRNN, stoi, itos, seed: str = "t", max_len: int = 20) -> str:
    """Generate one name character by character, starting from a seed.

    Args:
        model: the trained CharRNN.
        stoi: character -> index, for encoding the seed.
        itos: index -> character, for decoding what the model draws.
        seed: the starting character(s). They appear in the output.
        max_len: how many characters to add before giving up. A model that
            never draws END must still terminate.

    Returns:
        The generated name INCLUDING the seed and EXCLUDING the END marker.
    """
    model.eval()
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   start the result off as the seed's characters
    #   encode the seed as a (1, seq_len) tensor and run it through the model
    #       once, keeping both the scores and the hidden state
    #   then, up to max_len times:
    #       take the scores at the LAST position only
    #       turn them into probabilities and draw ONE character from that
    #           distribution
    #       stop as soon as you draw the END marker
    #       otherwise add the character to the result, and run just that one
    #           character back through the model, passing the hidden state in
    #   join the result into a string
    #
    #   Draw from the distribution, do not take the most likely character, or
    #   every name this model generates will be the same name.
    #
    raise NotImplementedError


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
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        raise SystemExit(main())
    except NotImplementedError:
        print("char_rnn.py is not finished yet: fill in the next TODO in this file, then re-run.")
