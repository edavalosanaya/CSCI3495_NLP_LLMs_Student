#!/usr/bin/env python3
"""W5C1 starter, additive (Bahdanau) attention from scratch + a text heatmap.

Run inside the course container:
    python weeks/week-05/class-01/exercise/attention.py

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-05/class-01/exercise/test_attention.py -k step1 -q

Tiny + CPU-only + deterministic.
"""
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
    """Return the score e_i for each key:  e_i = vᵀ tanh(W_s s + W_h h_i).

    Output shape: (num_keys,).
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #   sq = query @ W_s.T                  # (attn,)
    #   kh = keys @ W_h.T                   # (num_keys, attn)
    #   pre = torch.tanh(sq + kh)           # broadcast -> (num_keys, attn)
    #   return pre @ v                      # (num_keys,)
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
        """Return (context, weights).

        context: (hidden,) weighted blend of values.
        weights: (num_keys,) attention distribution that sums to 1.
        """
        # TODO (STEP 2): implement. Check with: pytest -k step2
        #   1) e = additive_scores(query, keys, self.W_s, self.W_h, self.v)
        #   2) weights = torch.softmax(e, dim=0)
        #   3) context = weights @ values        # (hidden,)
        #   return context, weights
        raise NotImplementedError


def heatmap(weights: torch.Tensor, row_labels=None, col_labels=None) -> str:
    """Render an attention matrix (num_rows, num_cols) as an ASCII shaded grid.

    Use the shade ramp below: higher weight -> denser character.
    """
    # GIVEN (STEP 3): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
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
    raise SystemExit(main())
