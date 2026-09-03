#!/usr/bin/env python3
"""W5C1 starter, additive (Bahdanau) attention from scratch + a text heatmap."""
from __future__ import annotations

import torch
import torch.nn as nn

SEED = 0


def additive_scores(
    query: torch.Tensor,     # (hidden,)         current decoder state s
    keys: torch.Tensor,      # (num_keys, hidden) encoder states h_i
    W_s: torch.Tensor,       # (attn, hidden)
    W_h: torch.Tensor,       # (attn, hidden)
    v: torch.Tensor,         # (attn,)
) -> torch.Tensor:
    """Score how well each key matches the query, Bahdanau style.

    Args:
        query: shape (hidden,). The decoder state doing the looking.
        keys: shape (num_keys, hidden). One encoder state per source position.
        W_s: shape (attn, hidden). Projects the query into the scoring space.
        W_h: shape (attn, hidden). Projects each key into the same space.
        v: shape (attn,). Collapses a scoring-space vector to one number.

    Returns:
        Shape (num_keys,): one raw score per key. Not yet a distribution, so
        the values may be negative and do not sum to anything in particular.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The score formula is in README section 3.
    #
    #   project the query with W_s, giving one vector of length attn
    #   project every key with W_h, giving one such vector per key
    #   add the query's projection to each key's, squash with tanh
    #   collapse each row to a single number with v
    #
    #   The query is added to EVERY row: one projected query broadcasts across
    #   all the keys, which is why no loop is needed.
    #
    raise NotImplementedError


class AdditiveAttention(nn.Module):
    def __init__(self, hidden: int, attn: int = 16):
        super().__init__()
        self.W_s = nn.Parameter(torch.randn(attn, hidden) * 0.1)
        self.W_h = nn.Parameter(torch.randn(attn, hidden) * 0.1)
        self.v = nn.Parameter(torch.randn(attn) * 0.1)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend over the values and report where the attention went.

        Args:
            query: shape (hidden,). The state doing the looking.
            keys: shape (num_keys, hidden). What the query is matched against.
            values: shape (num_keys, hidden). What gets blended. Same number of
                rows as keys, because row i of each describes position i.

        Returns:
            (context, weights). context is (hidden,), one blended vector.
            weights is (num_keys,) and sums to 1, and is returned so the
            attention can be inspected rather than just used.
        """
        # TODO (STEP 2): implement. Check with: pytest -k step2
        #
        #   Three lines: the two formulas in README section 4, in order.
        #
        #   score every key against the query, using this module's parameters
        #   turn those scores into a distribution over the keys
        #   blend the values by those weights
        #   return the blend and the weights, in that order
        #
        #   Softmax over the key axis, not over the hidden axis: you want one
        #   number per key that all sum to 1.
        #
        raise NotImplementedError


def heatmap(weights: torch.Tensor, row_labels=None, col_labels=None) -> str:
    """GIVEN. Renders an attention matrix as an ASCII shaded grid."""
    ramp = " .:-=+*#%@"
    W = weights.detach()
    if W.ndim == 1:
        W = W.unsqueeze(0)
    rows, cols = W.shape
    lines = []
    if col_labels:
        lines.append("      " + " ".join(f"{c[:4]:>4}" for c in col_labels))
    for r in range(rows):
        label = (row_labels[r] if row_labels else f"q{r}")[:5]
        cells = []
        for c in range(cols):
            level = int(round(float(W[r, c]) * (len(ramp) - 1)))
            level = max(0, min(len(ramp) - 1, level))
            cells.append(ramp[level] * 4)
        lines.append(f"{label:>5} " + " ".join(cells))
    return "\n".join(lines)


def align_briefly(attn: nn.Module, qs, keys, values, steps: int = 300) -> None:
    """(Given.) Train the scorer so query i attends to key i.

    A freshly initialized scorer has no opinion: every weight comes out ~1/3
    and the heatmap is a uniform gray blur. Attention weights are LEARNED;
    a few hundred tiny gradient steps make the diagonal emerge.
    """
    opt = torch.optim.Adam(attn.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.zeros(())
        for i, q in enumerate(qs):
            _, w = attn(q, keys, values)
            loss = loss - torch.log(w[i] + 1e-9)
        loss.backward()
        opt.step()


def main() -> int:
    torch.manual_seed(SEED)
    hidden = 8
    # Three encoder states; make them distinct so attention can separate them.
    keys = torch.eye(3, hidden)              # (3, 8): one-hot-ish encoder states
    values = keys.clone()
    attn = AdditiveAttention(hidden)
    qs = torch.eye(3, hidden)                # three queries, toward key 0, 1, 2

    query = qs[1]
    context, weights = attn(query, keys, values)
    weights = weights.detach()
    print("UNTRAINED attention weights:", [round(float(w), 3) for w in weights])
    print("  (a random scorer has no opinion yet: everything gets ~1/3)")
    print("weights sum:", round(float(weights.sum()), 4))
    print()
    print("Untrained heatmap (3 queries x 3 keys), a uniform gray blur:")
    rows = torch.stack([attn(q, keys, values)[1] for q in qs])
    print(heatmap(rows, row_labels=["q0", "q1", "q2"], col_labels=["k0", "k1", "k2"]))
    print()

    align_briefly(attn, qs, keys, values)

    context, weights = attn(query, keys, values)
    context, weights = context.detach(), weights.detach()
    print("TRAINED attention weights (query 1):", [round(float(w), 3) for w in weights])
    print("context vector:", [round(float(x), 3) for x in context])
    print()
    print("Trained heatmap, the diagonal emerges (dark = high weight):")
    rows = torch.stack([attn(q, keys, values)[1] for q in qs])
    print(heatmap(rows, row_labels=["q0", "q1", "q2"], col_labels=["k0", "k1", "k2"]))
    return 0


if __name__ == "__main__":
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
        raise SystemExit(main())
    except NotImplementedError:
        print("attention.py is not finished yet: fill in the next TODO in this file, then re-run.")
