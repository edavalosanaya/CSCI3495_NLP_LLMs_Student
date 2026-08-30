"""W9C1 starter, LoRA fine-tuning + quantization bake-off (CPU, tiny, fast).

Two ideas, hands-on, in pure PyTorch (no downloads):
  Part A: implement a LoRA adapter on a frozen Linear layer and fine-tune it on
          a toy task. Only the low-rank A, B train; the base weight is frozen.
  Part B: quantize a tensor of weights to k bits and measure the error vs.
          memory saved -- the quantization "bake-off".

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-09/class-01/exercise/test_lora_lab.py -k step1 -q

When all five steps are done, the demo runs:
    python weeks/week-09/class-01/exercise/lora_lab.py

Everything is tiny and seeded so it finishes in well under a minute on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank update  B @ A  (scaled).

    Output:  y = x @ W^T  +  scaling * (x @ A^T @ B^T)
    - The base weight `linear.weight` is FROZEN (requires_grad = False).
    - Only `A` (r x in) and `B` (out x r) are trainable.
    - Initialize A ~ small random, B = 0 so training starts at the base model.
    - scaling = alpha / r.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # TODO (STEP 1): freeze the base weight (requires_grad = False).
        #                Check with: pytest -k step1
        # TODO (STEP 2): create trainable A (r x in_features) and
        #                B (out_features x r). Init A with small randn, B zeros.
        #                Check with: pytest -k step2
        # TODO (STEP 3): store scaling = alpha / r. Check with: pytest -k step3
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (STEP 4): return base output + scaling * low-rank update.
        #                Bracket as (x @ A.T) @ B.T, not x @ (B @ A).
        #                Check with: pytest -k step4
        raise NotImplementedError


def train_lora(model: LoRALinear, X: torch.Tensor, Y: torch.Tensor, steps: int = 200):
    """Fit ONLY the LoRA params to regress Y from X (MSE). Return loss history."""
    # Optimize only parameters that require grad (i.e. A and B).
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
    """Symmetric per-tensor quantize-dequantize to `bits` integer levels.

    Steps:
      1. scale = max(abs(w)) / (2**(bits-1) - 1)
      2. q = round(w / scale), clamped to [-(2**(bits-1)-1), 2**(bits-1)-1]
      3. return q * scale  (the dequantized approximation, same shape/dtype)
    Returns w unchanged behavior at high bit-depth; coarse at low bit-depth.
    """
    # GIVEN (STEP 5): written for you. Read it, run its check, and use
    # it as the pattern for the steps you do write.
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return q * scale


def quant_error(w: torch.Tensor, bits: int) -> float:
    """Mean absolute error between original and quantized weights."""
    return (w - quantize(w, bits)).abs().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    # --- Part A: LoRA fine-tune on a toy linear task ---
    in_f, out_f, n = 8, 4, 64
    X = torch.randn(n, in_f)
    true_W = torch.randn(out_f, in_f)
    Y = X @ true_W.T  # target the adapter must help fit

    model = LoRALinear(in_f, out_f, r=4, alpha=8)
    losses = train_lora(model, X, Y, steps=200)
    print(f"LoRA loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable}  |  Frozen params: {frozen}")

    # --- Part B: quantization bake-off ---
    w = torch.randn(1000)
    print("\nQuantization bake-off (mean abs error vs original):")
    for bits in (8, 4, 2):
        print(f"  {bits}-bit: error = {quant_error(w, bits):.4f}")
