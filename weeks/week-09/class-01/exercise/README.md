# W9C1 Lab: LoRA & Quantization

## 1. Learning objective

Fine-tune a frozen layer by training a tiny low-rank adapter beside it, then
see what quantizing the weights costs in accuracy.

You write two things in `lora_lab.py`: the adapter's constructor and its
forward pass. The training loop and the quantizer are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-09/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `LoRALinear.__init__`

![LoRA: freeze W, train a low-rank B @ A update, merge at inference (Hu et al. 2021, Fig. 1)](../lecture/visuals/assets/lora-2021-fig-1.png)

$W_0$ never moves. The update is forced through $r$ dimensions, so it costs
$r(d_{in} + d_{out})$ parameters instead of $d_{in} d_{out}$:

$$A \in \mathbb{R}^{r \times d_{in}}, \qquad B \in \mathbb{R}^{d_{out} \times r}, \qquad r \ll \min(d_{in}, d_{out})$$

$B$ starts at zero, so at step 0 the adapter contributes nothing and training
begins from the pretrained model rather than a randomly damaged one.

Freeze the base weight, add a down-projection and an up-projection, and store
the scaling. The up-projection starts at zero.

```bash
pytest -k step1 -q
```

```
...                                                                      [100%]
3 passed, 6 deselected
```

## 4. Implement `LoRALinear.forward`

The frozen layer and the adapter are added, with the adapter scaled by
$\alpha / r$:

$$h = W_0 x + \frac{\alpha}{r}\, B A x$$

Base output, plus the scaled down-then-up update.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 7 deselected
```

## 5. Run it, then break it

Quantization is a separate saving, and the given `quantize` stores each weight
on a coarse grid:

$$s = \frac{\max |w|}{2^{k-1} - 1}, \qquad \hat{w} = s \cdot \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{w}{s}\right), -(2^{k-1}-1), \, 2^{k-1}-1\right)$$

```bash
python lora_lab.py
```

```
LoRA loss: 9.744 -> 0.000
Trainable params: 48  |  Frozen params: 32

Quantization bake-off (mean abs error vs original):
  8-bit: error = 0.0072
  4-bit: error = 0.1316
  2-bit: error = 0.7530
```

Note that here the adapter has MORE parameters than the layer it is adapting.
Each experiment below is a one-line edit; undo it before the next.

1. Sweep the rank. Build the adapter with `r=` 1, 2, 4 and 8. Trainable
   parameters go 12, 24, 48, 96 while frozen stays 32. On an 8x4 layer, LoRA is
   a loss, not a saving. At what layer size does it start paying off, and why
   are real LLM layers on the right side of that line?
2. Start `B` at random instead of zero. Set `self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)`.
   The adapter's output at initialization is no longer identical to the frozen
   layer's. What does that cost you at the very start of fine-tuning?
3. Quantize harder. The error climbs 0.0072, 0.1316, 0.7530 for 8, 4 and 2 bits,
   roughly ten times worse per halving. Which of those would you still ship, and
   what would you need to measure to decide properly?
4. Quantize the adapter too. Apply `quantize` to `A` and `B` rather than to the
   base weight. Is the damage comparable, and does that suggest anything about
   which parts of a model are safe to compress?
