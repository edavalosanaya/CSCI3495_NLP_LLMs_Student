# W4C1 Walkthrough: MLP text classifier, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `mlp_classifier.py` in this folder. Every code block below
is taken from it, and every printed value was produced by running it.

---

## Orientation

`build_vocab` ships written. Two details in it matter downstream:

```python
    vocab: dict[str, int] = {"<unk>": 0}
```

Id **0 is reserved** before any real word is seen, and `embed_document` uses
`vocab.get(tok, 0)`, so any out-of-vocabulary token silently becomes `<unk>`.
With 25 words in the whole vocabulary, that happens constantly the moment a
student types their own review. Worth demonstrating: feed the trained model a
sentence of entirely unseen words and watch it produce a confident prediction
from nothing but the `<unk>` embedding.

---

## Step 1, `embed_document`

```python
def embed_document(text: str, vocab: dict[str, int], emb: nn.Embedding) -> torch.Tensor:
    ids = []
    for tok in tokenize(text):
        # Row 0 is the unknown-word row, used for any token not in the vocab.
        ids.append(vocab.get(tok, 0))

    if len(ids) == 0:
        # No tokens at all, but the caller still needs a vector of the right
        # width, so hand back zeros rather than averaging an empty list.
        return torch.zeros(emb.embedding_dim)

    idx = torch.tensor(ids, dtype=torch.long)
    vectors = emb(idx)              # (num_tokens, embedding_dim)
    return vectors.mean(dim=0)      # average down to (embedding_dim,)
```

**`dim=0` is the entire step.** `vectors` is `(num_tokens, 16)`. Averaging over
dim 0 collapses the tokens and keeps the 16 features, giving `(16,)`. Averaging
over dim 1 would collapse the features and keep the tokens, giving
`(num_tokens,)`, which is wrong but does not raise, and the error only surfaces
later as a shape mismatch inside `fc1`. Students who get a confusing "mat1 and
mat2 shapes cannot be multiplied" in Step 2 almost always have this backwards.

**The `if not ids` guard above it** returns a zero vector for an empty document.
Without it, `mean` over an empty tensor returns `nan`, which then propagates
through the loss and turns the whole training run into `nan` with no obvious
cause.

**What you should see:**

```python
>>> d = embed_document("i loved this movie it was great", vocab, emb)
>>> tuple(d.shape)
(16,)
```

**The conceptual cost, worth stating explicitly.** Averaging is a bag-of-words
operation on embeddings. "good not bad" and "bad not good" contain the same
tokens, so they produce the *identical* vector and are therefore
*indistinguishable* to everything downstream. No amount of training fixes this;
it is thrown away before the model sees it. This is the concrete motivation for
sequence models (W4C2) and attention (W5).

---

## Step 2, `MLP.forward`

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
```

**Read it inside out:** `fc1` projects 16 down to 8, `act` (ReLU) zeroes the
negatives, `fc2` projects 8 to 2. The return value is two **logits**.

**No softmax here, deliberately.** `nn.CrossEntropyLoss` expects raw logits and
applies `log_softmax` internally. Adding a softmax in `forward` means the loss
softmaxes an already-softmaxed vector; training still "works" but converges much
more slowly and the gradients are wrong. It is one of the most common PyTorch
bugs and it never raises an error, so it is worth pre-empting out loud.

**Why the ReLU is load-bearing.** Without it, `fc2(fc1(x))` is a product of two
matrices, which is just another matrix. The network would be exactly equivalent
to a single linear layer and could only separate classes with a straight line.
The nonlinearity is the only reason "deep" buys anything.

**What you should see:**

```python
>>> tuple(model(torch.stack([d, d])).shape)
(2, 2)
```

---

## Given, the training loop

```python
    for _ in range(epochs):
        optimizer.zero_grad()
        # Re-embed each step so gradients flow into the embedding table too.
        X = _embed_batch(data, vocab, emb)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        history.append(float(loss))
```

**The five lines, and what breaks without each.**

- `optimizer.zero_grad()`: PyTorch **accumulates** gradients into `.grad` by
  default. Skip this and epoch 10's update includes the summed gradients of
  epochs 1 through 10. Training appears to explode. This is the single most
  common bug in the exercise.
- `logits = model(X)`: the forward pass, which also builds the autograd graph.
- `loss = loss_fn(logits, y)`: cross-entropy over the two classes.
- `loss.backward()`: walks that graph backwards, filling `.grad` on every
  parameter.
- `optimizer.step()`: applies the update. Calling it before `backward()` steps on
  stale or absent gradients.

**Why `X` is rebuilt inside the loop.** The embedding table is being trained too
(`params = list(model.parameters()) + list(emb.parameters())`). Computing `X`
once outside the loop would detach it from the current embedding weights, so
gradients would never reach `emb` and the word vectors would stay at their random
initialization. The model would still learn something, using the MLP alone on
fixed random features, which is a subtly worse result that no test catches. Point
at this line when a student asks why it is not hoisted out for speed.

**`float(loss)`** on the last line converts the tensor to a plain Python float,
which also detaches it from the graph so `history` does not keep the entire
computation graph alive. It emits a `UserWarning` about converting a tensor with
`requires_grad=True`; harmless here, and `loss.item()` or `loss.detach()` would
silence it.

---

## Running it

```
vocab size: 25
epoch   1 loss: 0.7414
epoch 200 loss: 0.0000
train accuracy: 1.00
```

**Two numbers to teach from.**

1. **Loss starts at 0.7414.** For two balanced classes, a model that knows
   nothing outputs 50/50, and the cross-entropy of that is
   $-\ln(0.5) = 0.693$. Starting near that value confirms the network really is
   initialized at chance, and it is a sanity check students can reuse: if a fresh
   binary classifier starts far from 0.693, something is already wrong.

2. **Loss ends at 0.0000 with 100% accuracy, and that is not good news.** Eight
   training documents, no held-out set, and enough parameters to memorize every
   one of them. The model has learned these eight sentences, not sentiment. Ask
   the class what result would actually be evidence of learning; the answer,
   accuracy on data it has never seen, is the whole reason every later exercise
   carries a test split.

The deliberately separable dataset makes the mechanics visible, which is the
point. Just do not let the perfect score go unremarked, or students draw exactly
the wrong lesson from it.
