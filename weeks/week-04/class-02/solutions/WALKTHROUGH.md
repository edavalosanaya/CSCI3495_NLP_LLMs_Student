# W4C2 Walkthrough: Char-level RNN, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `char_rnn.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it at the default
seed (`SEED = 1`).

---

## Step 0, Orientation

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

## Step 1, The forward pass

```python
    def forward(self, ids: torch.Tensor, h0: torch.Tensor | None = None):
        x = self.emb(ids)
        out, h_n = self.rnn(x, h0)
        logits = self.out(out)
        return logits, h_n
```

**Shapes, which is where students get lost.** `ids` is `(batch, seq)`. After
`emb` it is `(batch, seq, emb_dim)`. `nn.RNN` with `batch_first=True` returns
`out` of shape `(batch, seq, hidden)`, the hidden state at **every** timestep,
plus `h_n`, just the final one. The `out` linear layer maps `hidden` to
`vocab_size` at every position, giving `(batch, seq, vocab_size)`.

**Returning `h_n` is not optional.** `sample` feeds one character at a time and
must pass the previous hidden state back in; a `forward` that returns only logits
makes Step 3 unimplementable. Students who return a bare tensor hit a confusing
unpacking error in `train` before they ever reach sampling.

**`h0=None` is meaningful.** PyTorch treats it as a zero initial hidden state, so
the first call in `sample` (which passes no `h0`) starts from a blank memory,
while subsequent calls thread the real state through.

**What you should see:**

```python
>>> ids = torch.tensor([[stoi[c] for c in "abc"]])
>>> logits, h = model(ids)
>>> tuple(logits.shape)
(1, 3, 23)
```

Three input characters produce three full distributions, not one. Every position
is a training example, which is what makes such a tiny corpus trainable at all.

---

## Step 2, Build the training pairs

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

## Step 3, Sample a new name

```python
    for _ in range(max_len):
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1))
        ch = itos[nxt]
        if ch == END:
            break
        result.append(ch)
        ids = torch.tensor([[nxt]], dtype=torch.long)
        logits, h = model(ids, h)
    return "".join(result)
```

**Four things happen per iteration**, and three of them are common failure
points.

1. `logits[0, -1]` takes the **last** position's prediction. On the first pass
   the seed may be several characters long, and only the final position's
   distribution predicts the next character.
2. `torch.multinomial` **samples** rather than taking the argmax. Argmax would
   make every seed produce exactly one name forever; sampling is what makes the
   activity interesting. This is the greedy-vs-sampling distinction Week 7
   develops properly.
3. `break` on `END` before appending, so the marker never lands in the output
   string. The test asserts `END not in name`.
4. `logits, h = model(ids, h)` **passes `h` back in**. Omitting it is the classic
   bug: generation still runs and still produces letters, but each character is
   drawn with no memory of what came before, so the output degenerates into
   plausible-looking noise. Because it does not crash, students can lose real
   time here. If a name looks like random letters even after training, check
   this line first.

**`@torch.no_grad()` and `model.eval()`** on the function keep autograd off and
put any dropout or batchnorm into inference mode. Neither matters for this tiny
model, but the habit is worth building.

---

## Step 4, Train and generate

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

Useful diagnostic to run live: cut `epochs` to 100 and re-run. The loss is
higher, the names are worse letter-by-letter, but a larger fraction of them are
genuinely new. That trade-off between fitting the data and inventing something is
worth ten minutes of discussion.
