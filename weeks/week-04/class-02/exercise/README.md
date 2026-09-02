# W4C2 Lab: Character RNN Name Generator

## 1. Learning objective

Train a recurrent network one character at a time, then sample from it to
invent new dinosaur names, and see where the invention actually comes from.

You write two short functions in `char_rnn.py`, one for each formula in the
next section. The model, the training loop and the generation loop are given.

## 2. Understanding the math

![RNN unrolled over four time steps with shared weights](../lecture/visuals/rnn-unrolled.png)

An RNN carries a hidden state $h_t$ forward through the sequence, updating it
from the previous state and the current character. The same weights are reused
at every step, which is why the network handles names of any length:

$$h_t = \tanh(W_h\, h_{t-1} + W_x\, x_t + b)$$

$h_{t-1}$ is the hidden state carried in, $x_t$ is the current character after
the embedding layer, and $W_h$, $W_x$ and $b$ are the weights. Every product is
a matrix times a vector, which is `@` in PyTorch, and $\tanh$ is `torch.tanh`.

![RNN language model: softmax over the vocabulary, sampled token fed back in](../lecture/visuals/rnn-lm.png)

The model turns each hidden state into one raw score per vocabulary character.
Softmax makes those scores a probability distribution over what comes next:

$$P(x_{t+1} = c \mid x_1 \ldots x_t) = \mathrm{softmax}(z_t)_c$$

Generating means running that loop forward: score the next character, draw one
from that distribution, feed it back in as the next input. Drawing is
`torch.multinomial`, which picks an index with probability equal to its weight.

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-04/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `rnn_step`

One line: the first formula above, for a single character. `nn.RNN` is this
formula run in a loop, so the demo checks your version against it and the two
should agree to about `1e-07`.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 5 deselected
```

## 5. Implement `sample_next`

Three lines: take the model's scores at the last position, softmax them into
probabilities, and draw one index. The loop that calls it is given.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 5 deselected
```

## 6. Run it, then break it

```bash
python char_rnn.py
```

```
(seed 1: same names as everyone else; rerun with --seed <n> for your own)
one step of the recurrence, nn.RNN vs your rnn_step:
  nn.RNN   [0.9934345483779907, 0.9953884482383728, -0.880752444267273, 0.9822033643722534]
  rnn_step [0.9934345483779907, 0.9953884482383728, -0.8807525038719177, 0.9822033643722534]
  max difference: 1.19e-07

vocab size: 23
epoch   1 loss: 3.1450
epoch 400 loss: 0.0916

Generated dinosaur names:
  tyrannosaurus
  allosaurus
  spinosaurus
  velociraptor
  megalosaurus
  rantyrannosaurus
```

Five of those six are already in `NAMES`. Each experiment below is a one-line
edit; undo it before the next.

1. Take the most likely character instead of drawing one. In `sample_next`,
   replace the `torch.multinomial` call with `torch.argmax(probs)`. The six
   names become `triceratops`, `allosaurus`, `stegosaurus`, `velociraptor`,
   `maiasaura`, `raptor`, and they are the same six every run. Which behaviour
   do you want from a name generator, and which from a translator?
2. Train less. Change `epochs: int = 400` to `epochs: int = 20`. Loss stops at
   1.0677 and the names become `tarnodachalosaurus`, `aprus`, `saurus`,
   `veredtorathatachachys`, `mus`, `rychzs`. These are far more original than
   the fully-trained model's. Is the trained model better or worse at this
   task, and what does that tell you about what the loss is measuring?
3. Seed a letter the data never starts with. Change the seed list in `main` to
   `["z"]`. It produces `zinosaurus`, even though no training name begins with
   `z`. Where could that possibly come from, given the model has never seen
   such a name?
4. Count the memorization. Change the seed list to all 13 starting letters in
   `NAMES`: `["a", "b", "c", "d", "g", "i", "m", "o", "p", "s", "t", "u", "v"]`.
   All 13 generated names are exact training names. What would you change about
   the setup to get names that are new but still dinosaur-shaped?
