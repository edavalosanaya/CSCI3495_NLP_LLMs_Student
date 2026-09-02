# W4C2 Lab: Character RNN Name Generator

## 1. Learning objective

Train a recurrent network one character at a time, then sample from it to
invent new dinosaur names, and see where the invention actually comes from.

You write two things in `char_rnn.py`: the forward pass, and the sampling
loop. The vocabulary, the training pairs and the training loop are given.

## 2. Understanding the math

![RNN unrolled over four time steps with shared weights](../lecture/visuals/rnn-unrolled.png)

An RNN carries a hidden state $h_t$ forward through the sequence, updating it
from the previous state and the current character. The same weights are reused
at every step, which is why the network handles names of any length:

$$h_t = g(W_h\, h_{t-1} + W_x\, x_t + b) \qquad P(x_t \mid x_1 \ldots x_{t-1}) = \mathrm{softmax}(W_o\, h_t)$$

![RNN language model: softmax over the vocabulary, sampled token fed back in](../lecture/visuals/rnn-lm.png)

Generating means running that loop forward: score the next character, draw one
from the distribution, feed it back in as the next input.

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-04/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `CharRNN.forward`

Embed, run the RNN, project to one score per character. Return the scores and
the final hidden state, because the sampler feeds that state back in.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 4 deselected
```

## 5. Implement `sample`

Run the loop forward one character at a time, drawing from the distribution
rather than taking the most likely character.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 4 deselected
```

## 6. Run it, then break it

```bash
python char_rnn.py
```

```
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

1. Take the most likely character instead of drawing one. Replace the
   `torch.multinomial` call with `torch.argmax(probs)`. Run it twice with the
   same seed letter: you now get the identical name both times, where sampling
   gave `triceratops` once and `therizinosaurus` the next. Which behaviour do
   you want from a name generator, and which from a translator?
2. Train less. Change `epochs=400` to `epochs=20`. Loss stops at 1.0252 and the
   names become `toxophesadonyoxurus`, `alrilosaurus`, `saulovirastylasaurody`.
   These are far more original than the fully-trained model's. Is the trained
   model better or worse at this task, and what does that tell you about what
   the loss is actually measuring?
3. Seed a letter the data never starts with. Call `sample(..., seed="z")`. It
   produces `ziniovigus` even though no training name begins with `z`. Where
   could that possibly come from, given the model has never seen such a name?
4. Count the memorization. With the full 400 epochs, every one of the 13
   distinct starting letters in `NAMES` generates an exact training name. What
   would you change about the setup to get names that are new but still
   dinosaur-shaped?
