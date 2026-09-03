"""W9C1 starter, LoRA fine-tuning + quantization bake-off (CPU, tiny, fast)."""
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
        """Set up a frozen base layer plus a trainable low-rank adapter.

        Args:
            in_features: width of the input.
            out_features: width of the output.
            r: the adapter's rank. This is the whole point: the update is
                forced through r dimensions, so it has r*(in+out) parameters
                instead of in*out.
            alpha: scaling numerator. The adapter is scaled by alpha/r so that
                changing r does not silently change how strong it is.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # TODO (STEP 1): implement. Check with: pytest -k step1
        #
        #   The shapes are in README section 3, the scaling in section 4.
        #
        #   keep the Linear above, but freeze its weight so no gradient ever
        #       reaches it: that frozen matrix is the pretrained model
        #   make two trainable parameters, one going DOWN from in_features to
        #       r, and one coming back UP from r to out_features
        #   start the down-projection small and random, and the up-projection
        #       at exactly ZERO
        #   store the scaling factor from the formula
        #
        #   Zero on the up-projection is not an arbitrary choice: it makes the
        #   adapter contribute nothing at step 0, so training begins from the
        #   pretrained model rather than from a randomly damaged one.
        #
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the low-rank update to the frozen layer's output.

        Args:
            x: shape (batch, in_features).

        Returns:
            Shape (batch, out_features). Identical to the frozen layer's own
            output until the adapter has been trained.
        """
        # TODO (STEP 2): implement. Check with: pytest -k step2
        #
        #   run x through the frozen layer to get the base output
        #   separately, send x DOWN through the rank-r projection and then back
        #       UP, which is the whole low-rank update
        #   add the update to the base, scaled by the stored factor
        #
        #   Down first, then up. Doing it in one big matrix would work
        #   numerically and defeat the entire purpose.
        #
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
    """GIVEN. Quantizes to `bits` levels and back, so you can see the error."""
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
    # A student running this before finishing should see a sentence, not a
    # traceback: an unwritten step is a normal state, not a crash.
    try:
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
    except NotImplementedError:
        print("lora_lab.py is not finished yet: fill in the next TODO in this file, then re-run.")
