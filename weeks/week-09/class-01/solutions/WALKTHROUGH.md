# W9C1 Walkthrough: LoRA and quantization, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `lora_lab.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it.

---

## Step 1, `LoRALinear.__init__`

```python
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # freeze the base weight
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
```

**Step 1, the freeze.** One line, and it is the premise of the entire
parameter-efficient fine-tuning literature. Note it is set on
`self.linear.weight`, not on the module: PyTorch tracks `requires_grad`
per-tensor. `train_lora` then filters with
`[p for p in model.parameters() if p.requires_grad]`, so the frozen weight never
even reaches the optimizer.

**Step 2, the asymmetric initialization.** `A` small random, `B` exactly zero.
Students reliably ask why not the reverse, and the answer is a nice bit of
calculus:

$$\frac{\partial}{\partial B}\big[(BA)x\big] \propto Ax$$

With `A = 0` that gradient is zero, so `B` could never leave zero and the adapter
would be permanently dead. With `B = 0` and `A` nonzero, the *product* is still
zero (so the model starts at the base) but the gradient reaching `B` is not. Only
one of the two orderings works. This is Hu et al.'s choice and the reason is
exactly this.

**Step 3, the scaling.** `alpha / r` decouples the update's magnitude from the
rank. Without it, doubling `r` roughly doubles the size of the update (more terms
in the sum), and every hyperparameter would need retuning whenever the rank
changed. With it, `alpha` controls strength and `r` controls capacity, which is
why papers report them as separate knobs.

---

## Step 2, `LoRALinear.forward`

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        update = (x @ self.A.T) @ self.B.T
        return base + self.scaling * update
```

**The bracketing is the performance story.** Both of these compute the same
thing:

- `(x @ A.T) @ B.T`: project into rank `r`, then back out. Cost `O(n·d·r)`.
- `x @ (B @ A)`: build the full `d_out × d_in` update first. Cost dominated by
  materializing a matrix as big as the frozen weight.

The second defeats the point of LoRA. On this toy layer neither is measurable,
but the habit matters, and it is worth showing the class the shapes to make it
concrete.

**What you should see:**

```python
>>> torch.manual_seed(0)
>>> m = LoRALinear(8, 4, r=4, alpha=8)
>>> x = torch.randn(2, 8)
>>> torch.allclose(m(x), m.linear(x))
True
```

**A fresh LoRA layer is a no-op.** That `True` is the practical consequence of
`B = 0` and it is a genuinely useful property: you can wrap every Linear in a
pretrained model with LoRA adapters and the model's behavior is bit-identical
until you train. No risk of degrading the base model by attaching adapters.

---

## Given, `quantize`

```python
def quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return q * scale
```

**"Symmetric per-tensor" unpacks into three choices**, each worth naming:

- **Symmetric**: the grid is centered on zero, running `-qmax` to `+qmax`. Simpler
  than asymmetric (which also learns a zero-point offset) and a good fit for
  weights, which are roughly zero-centered. Activations often are not, which is
  why activation quantization usually is asymmetric.
- **Per-tensor**: one `scale` for the whole tensor. Per-channel scales fit better
  and are standard in practice; per-tensor is simpler to read.
- **Quantize-dequantize**: it returns floats snapped to the representable grid,
  not integers. This is *simulated* quantization, which is how the error is
  measured without changing any downstream code. The real memory saving comes
  from storing `q` as int4/int8 plus one float scale, which this toy does not do.

**`if scale == 0`** handles an all-zero tensor, where the division would produce
`nan` and silently poison everything after it.

---

## Running it

```
LoRA loss: 9.744 -> 0.000
Trainable params: 48  |  Frozen params: 32

Quantization bake-off (mean abs error vs original):
  8-bit: error = 0.0072
  4-bit: error = 0.1316
  2-bit: error = 0.7530
```

**The error curve is the lesson, and it is geometric, not linear.** 8 to 4 bits
multiplies the error by roughly 18; 4 to 2 by roughly another 6. Each bit removed
halves the number of representable levels, so the spacing of the grid doubles and
the rounding error with it. (The exact ratios are not clean powers of two because
`qmax = 2^(k-1) - 1`: 127, 7, 1. Going from 7 levels to 1 is proportionally far
worse than 127 to 7.)

Draw the practical conclusion: **4-bit sits at the knee.** You get 8x memory
reduction versus fp32 for an error most models tolerate after fine-tuning, which
is why QLoRA (Dettmers et al. 2023) is built on 4-bit and why 2-bit remains a
research problem rather than a default.

**The trainable-parameter count needs an explicit caveat**, or students will draw
the wrong conclusion. This toy reports 48 trainable versus 32 frozen: LoRA
appears to have *increased* the parameter count. That is real, and it is an
artifact of the scale. With `in=8, out=4, r=4`, the factors `A` (4x8 = 32) and
`B` (4x4 = 16) total 48, while the base weight is only 32. Low-rank
approximation only saves anything when `r ≪ min(d_in, d_out)`, and here `r` is as
large as the output dimension.

Say the real numbers out loud for contrast: for GPT-3 175B with `r = 4` on the
attention projections, LoRA trains about 10,000x fewer parameters than full
fine-tuning (Hu et al. 2021, and the lecture's Fig. 2). The mechanism is what
this lab teaches; the ratio needs a real model to appear.

**The merge stretch goal is the one worth doing in class if there is time.**
Compute `W + scaling * (B @ A)`, load it into a plain `nn.Linear`, and confirm
identical outputs. That is why LoRA adds **zero** inference latency: once
training is done, the adapter folds into the weight and disappears. It is also
why you can ship many task-specific adapters against one base model and swap them
per request, which is the deployment story behind the whole technique.
