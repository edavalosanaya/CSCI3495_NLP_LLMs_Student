# W4C2 Walkthrough: Char-level RNN, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `char_rnn.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it at the default
seed (`SEED = 1`).

---

## Orientation

`build_vocab` puts `END` (`.`) into the vocabulary alongside the letters:

```python
    chars = sorted(set("".join(names)) | {END})
```

**Why the end marker is a vocabulary entry** rather than a special case in the
code: the model must *learn when to stop*. If stopping were handled outside the
network, generation would need an external length rule, and names would all come
out the same length. Making `.` predictable like any other character means "the
name is over" is something the model has an opinion about, and `sample` just
watches for it.

`sorted(...)` matters too: it makes the id assignment deterministic across runs,
without which nothing else in the exercise would reproduce.

---

## Step 1, `rnn_step`

```python
    combined = w_h @ h_prev + w_x @ x_t + b
    return torch.tanh(combined)
```

This is the formula in README section 3 typed out, and that is the whole point
of the step: the recurrence is two matrix-vector products, an add, and a
squash. Students who have been told an RNN is complicated are usually surprised
it fits on two lines.

**The shapes decide the order of the products.** `w_h` is `(hidden, hidden)`
and `h_prev` is `(hidden,)`, so `w_h @ h_prev` is `(hidden,)`. `w_x` is
`(hidden, emb_dim)` and `x_t` is `(emb_dim,)`, so that product is `(hidden,)`
too. Writing either product backwards raises a shape error immediately, which
is the good kind of bug.

**Why `tanh` and not ReLU.** The state is fed back into itself at every step,
so an unbounded activation lets it grow without limit over a long sequence.
`tanh` pins every entry to `(-1, 1)`, which is the cheapest way to keep a
recurrence stable. Every entry of the returned state being inside that range is
a quick check that the line is right.

**Common mistake:** adding `b` inside one of the products, or once per product.
The formula has a single bias, added after both.

---

## Given, `CharRNN.forward`

```python
        x = self.emb(ids)
        out, h_n = self.rnn(x, h0)
        logits = self.out(out)
        return logits, h_n
```

Given, but worth reading, because it is `rnn_step` at scale. `ids` is
`(batch, seq)`; after `emb` it is `(batch, seq, emb_dim)`. `nn.RNN` with
`batch_first=True` returns `out` of shape `(batch, seq, hidden)`, the hidden
state at **every** timestep, plus `h_n`, just the final one. The `out` linear
layer maps `hidden` to `vocab_size` at every position.

**`nn.RNN` is a loop over your `rnn_step`.** That is not a metaphor, and the
demo proves it: `compare_one_step` pulls the layer's own weights out and runs
one character both ways.

```python
    w_h, w_x, b = rnn_weights(model)
    h_zero = torch.zeros(model.rnn.hidden_size)
    h_mine = rnn_step(h_zero, x[0, 0], w_h, w_x, b)
```

The two hidden states agree to about `1e-07`, which is float32 rounding, not a
difference in the math. The one wrinkle is the bias: `nn.RNN` keeps `bias_ih_l0`
and `bias_hh_l0` where the formula has one `b`, so `rnn_weights` adds them. They
are redundant parameters, kept for symmetry with the two weight matrices.

**`h0=None` is meaningful.** PyTorch treats it as a zero initial hidden state,
so the first call in `sample` starts from a blank memory while later calls
thread the real state through.

---

## Given, `make_training_pairs`

```python
def make_training_pairs(name: str, stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    in_chars = list(name)
    out_chars = list(name[1:]) + [END]
    xin = torch.tensor([stoi[c] for c in in_chars], dtype=torch.long)
    yt = torch.tensor([stoi[c] for c in out_chars], dtype=torch.long)
    return xin, yt
```

**The shift is the whole idea.** For `"abc"`:

| position | input | target |
|---|---|---|
| 0 | `a` | `b` |
| 1 | `b` | `c` |
| 2 | `c` | `.` |

Both tensors have the same length, so cross-entropy lines up position by
position. Appending `END` to the targets (rather than to the inputs) is what
teaches the model to predict the stop, without ever feeding it a `.` as input
during training.

**This is self-supervision in its clearest form.** Nobody labelled this data. The
text is both the input and the answer key, which is exactly why next-token
prediction scales to trillions of tokens of ordinary web text and why it is the
pretraining objective for every model in the rest of the course.

**What you should see:**

```python
>>> xin, yt = make_training_pairs("abc", stoi)
>>> [itos[i.item()] for i in xin]
['a', 'b', 'c']
>>> [itos[i.item()] for i in yt]
['b', 'c', '.']
```

---

## Step 2, `sample_next`

```python
    last_scores = logits[0, -1]
    probs = torch.softmax(last_scores, dim=-1)
    drawn = torch.multinomial(probs, num_samples=1)
    return int(drawn)
```

**`logits[0, -1]` takes the last position.** On the first call the seed may be
several characters long, and only the final position's scores predict the
character that comes next. Taking `logits[0, 0]` is the mistake to watch for:
it predicts the second character of the name, over and over.

**`torch.multinomial` draws, it does not maximize.** It picks an index with
probability equal to its weight, so a character with 0.6 of the mass comes up
about 60% of the time and the other 40% goes somewhere else. `torch.argmax`
would return the same index every call, and every seed would produce exactly
one name forever. This is the greedy-versus-sampling distinction that Week 7
develops properly, and experiment 1 in the README is worth doing.

**`int(...)` matters.** `multinomial` returns a tensor; the caller uses the
result as a dictionary key in `itos`, and a tensor is not the key `itos` has.

**Softmax before the draw, not after.** `multinomial` needs non-negative
weights, and raw logits are free to be negative.

---

## Given, `sample`

```python
    for _ in range(max_len):
        nxt = sample_next(logits)
        ch = itos[nxt]
        if ch == END:
            break
        result.append(ch)
        ids = torch.tensor([[nxt]], dtype=torch.long)
        logits, h = model(ids, h)
```

The loop is given because its bugs are not about language models. Two lines are
worth pointing at anyway:

- `break` on `END` happens **before** the append, so the marker never lands in
  the returned string.
- `logits, h = model(ids, h)` **passes `h` back in**. Dropping it is the classic
  RNN bug: generation still runs and still produces letters, but each character
  is drawn with no memory of what came before, so the output degenerates into
  plausible-looking noise. It does not crash, so it costs real time.

---

## Running it

```
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

**Start with the loss.** 3.1450 at epoch 1, and $\ln(23) = 3.135$ is the
cross-entropy of a uniform guess over the 23-character vocabulary. The untrained
model is at chance, exactly as it should be. This is the same sanity check as the
0.693 in W4C1, and students should be collecting these.

**Now the awkward part, which is the lesson.** Five of the six generated names
are **verbatim training examples**. Only `rantyrannosaurus` is novel, and it is a
splice of two memorized names. The model did not learn "what dinosaur names look
like"; with 34 examples and 400 epochs it learned the 34 examples.

Do not skip past this. It is the same overfitting students measured with the
trigram in W2C1, now in a neural model, and it sets up:

- **Why scale matters.** The reason GPT-style models generalize is not a cleverer
  architecture, it is that memorizing the training set stops being possible.
- **Why the name-vote activity is hard.** Finding a genuinely novel, plausible
  name means finding a spot where the model interpolated rather than copied.
  Students will need a few seeds. That difficulty *is* the finding.
- **The memorization thread**, which returns in W9C2 as training-data
  contamination and again in W15 as the question of whether a model reproducing
  its training data is quoting or creating.

**Running the activity.** The default seed is fixed so the tests are
deterministic and everyone's first run matches, which also means nobody has
"their own" names until they pass `--seed`. Say this explicitly before the vote,
or half the room will submit `tyrannosaurus`.

Useful diagnostic to run live: cut `epochs` to 20 and re-run. The loss stops at
1.0677, the names are worse letter-by-letter (`tarnodachalosaurus`, `aprus`,
`saurus`), but every one of them is new. That trade-off between fitting the data
and inventing something is worth ten minutes of discussion, and it is
experiment 2 in the README.
