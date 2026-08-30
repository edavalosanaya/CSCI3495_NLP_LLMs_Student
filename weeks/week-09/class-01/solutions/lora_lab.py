"""W9C1 reference solution, LoRA fine-tuning + quantization bake-off."""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # freeze the base weight
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        update = (x @ self.A.T) @ self.B.T
        return base + self.scaling * update


def train_lora(model: LoRALinear, X: torch.Tensor, Y: torch.Tensor, steps: int = 200):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=0.05)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return q * scale


def quant_error(w: torch.Tensor, bits: int) -> float:
    return (w - quantize(w, bits)).abs().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    in_f, out_f, n = 8, 4, 64
    X = torch.randn(n, in_f)
    true_W = torch.randn(out_f, in_f)
    Y = X @ true_W.T

    model = LoRALinear(in_f, out_f, r=4, alpha=8)
    losses = train_lora(model, X, Y, steps=200)
    print(f"LoRA loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable}  |  Frozen params: {frozen}")

    w = torch.randn(1000)
    print("\nQuantization bake-off (mean abs error vs original):")
    for bits in (8, 4, 2):
        print(f"  {bits}-bit: error = {quant_error(w, bits):.4f}")
