#!/usr/bin/env python3
"""W5C1 reference solution, additive (Bahdanau) attention + text heatmap."""
from __future__ import annotations

import torch
import torch.nn as nn

SEED = 0


def additive_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    W_s: torch.Tensor,
    W_h: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    sq = query @ W_s.T              # (attn,)
    kh = keys @ W_h.T              # (num_keys, attn)
    pre = torch.tanh(sq + kh)     # (num_keys, attn) via broadcast
    return pre @ v                # (num_keys,)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden: int, attn: int = 16):
        super().__init__()
        self.W_s = nn.Parameter(torch.randn(attn, hidden) * 0.1)
        self.W_h = nn.Parameter(torch.randn(attn, hidden) * 0.1)
        self.v = nn.Parameter(torch.randn(attn) * 0.1)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = additive_scores(query, keys, self.W_s, self.W_h, self.v)
        weights = torch.softmax(e, dim=0)
        context = weights @ values
        return context, weights


def heatmap(weights: torch.Tensor, row_labels=None, col_labels=None) -> str:
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
    """Train the scorer so query i attends to key i (maximize log alpha_ii)."""
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
    keys = torch.eye(3, hidden)
    values = keys.clone()
    attn = AdditiveAttention(hidden)
    qs = torch.eye(3, hidden)

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

    # sanity checks for the test sweep
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    assert context.shape == (hidden,)
    assert float(weights[1]) > 0.8, "trained attention should align q1 with k1"
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
